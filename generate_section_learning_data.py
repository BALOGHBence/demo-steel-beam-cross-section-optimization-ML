# Generate learning data for model training

from typing import Generator
import pandas as pd
import multiprocessing
import json, sys
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
import argparse
from argparse import Namespace
from pydantic import BaseModel

from cso.section import construct_section, section_properties, random_section_params
from cso.material import MaterialProperties, random_material_params
from cso.logger import get_logger, set_log_level
from cso.constants import CROSS_SECTION_PARAMETERS

load_dotenv()
logger = get_logger()
set_log_level("INFO")


class SampleInput(BaseModel):
    section_type: str
    section_params: dict
    mesh_sizes: int | list[int]
    material_params: dict


class Sample(BaseModel):
    section_type: str
    section_params: dict
    valid: bool
    stiffness_props: dict
    material_props: dict

    def as_flat_dict(self) -> dict:
        """Returns a flattened dictionary representation of the sample."""
        flat_dict = {
            "section_type": self.section_type,
            "valid": self.valid,
        }
        flat_dict.update({k: v for k, v in self.section_params.items()})
        flat_dict.update({k: v for k, v in self.stiffness_props.items()})
        flat_dict.update({k: v for k, v in self.material_props.items()})
        return flat_dict


def generate_sample(
    section_type: str, section_data: dict, material_params: dict
) -> Generator[SampleInput, None, None]:
    while True:
        section_params = random_section_params(section_data)
        material_params = random_material_params(material_params)
        mesh_sizes = section_data["mesh_sizes"]
        sample_input = SampleInput(
            section_type=section_type,
            section_params=section_params,
            mesh_sizes=mesh_sizes,
            material_params=material_params,
        )
        yield sample_input


def process_sample(input: SampleInput) -> dict:
    """Generate a single data sample of section parameters, loads, and utilization."""
    try:
        section = construct_section(
            geometry_constructor=input.section_type,
            params=input.section_params,
            material=input.material_params,
            mesh_sizes=input.mesh_sizes,
            calculate=True,
        )
        stiffness_props: dict = section_properties(section)
        valid = True
    except Exception as e:
        if "TopologyException" in str(e):
            logger.debug(f"TopologyException for params {input.section_params}: {e}")
        else:
            logger.error(
                f"Error generating sample for params {input.section_params}: {e}"
            )
        stiffness_props = {v: None for v in CROSS_SECTION_PARAMETERS}
        valid = False

    material_props = MaterialProperties(**input.material_params)
    relevant_material_props = {
        "elastic_modulus": material_props.elastic_modulus,
        "poissons_ratio": material_props.poissons_ratio,
        "yield_strength": material_props.yield_strength,
    }

    result = Sample(
        section_type=input.section_type,
        section_params=input.section_params,
        valid=valid,
        stiffness_props=stiffness_props,
        material_props=relevant_material_props,
    )

    return result


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Cross Section Optimization")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to the config file"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        help="Set logger level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=-1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for data generation",
    )
    parser.add_argument(
        "--output", type=str, default="data.csv", help="Output CSV file name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=500, help="Number of samples per batch"
    )
    return parser.parse_args()


def generate_learning_data():
    args = parse_args()
    set_log_level(args.loglevel)
    try:
        # Load configuration from JSON file
        with open(args.config, "r") as f:
            config = json.load(f)

        material_params = config["material"]
        section_data = config["section"]
        section_type = section_data["type"]

        sample_generator = generate_sample(section_type, section_data, material_params)

        if (num_samples := args.num_samples) < 0:
            num_samples = config["num_samples"]
        assert num_samples > 0, "Number of samples must be positive."

        batch_size = args.batch_size
        num_workers = args.num_workers
        output = args.output

        logger.info(f"Generating {num_samples} samples in batches of {batch_size}...")
        total_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_count = end_idx - start_idx
            logger.info(
                f"Processing batch {batch_idx+1}/{total_batches} ({batch_count} samples)..."
            )

            tasks = [next(sample_generator) for _ in range(batch_count)]
            data = []
            with multiprocessing.Pool(num_workers) as pool:
                with tqdm(
                    total=len(tasks), desc=f"Batch {batch_idx+1}/{total_batches}"
                ) as pbar:

                    def on_success(result: Sample) -> None:
                        data.append(result.as_flat_dict())
                        pbar.update(1)

                    def on_error(e: Exception) -> None:
                        logger.error(f"Error in worker: {e}")
                        pbar.update(1)

                    for t in tasks:
                        pool.apply_async(
                            process_sample,
                            args=(t,),
                            callback=on_success,
                            error_callback=on_error,
                        )
                    pool.close()
                    pool.join()

            df = pd.DataFrame(data)
            # Write header only for first batch
            write_header = batch_idx == 0
            df.to_csv(output, mode="a", header=write_header, index=False)
            logger.info(f"Appended batch {batch_idx+1} to {output}.")
        logger.info(f"Data generation completed. All batches saved to {output}.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(generate_learning_data())
