# Generate learning data for model training

from typing import Any
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


def generate_sample(input: SampleInput) -> dict:
    """Generate a single data sample of section parameters, loads, and utilization."""
    try:
        section = construct_section(
            geometry_constructor=input.section_type,
            params=input.section_params,
            material=input.material_params,
            mesh_sizes=input.mesh_sizes,
            calculate=True,
        )
        stiffness_props = section_properties(section)
        valid = True
    except Exception as e:
        if "TopologyException" in str(e):
            logger.debug(f"TopologyException for params {input.section_params}: {e}")
        else:
            logger.error(f"Error generating sample for params {input.section_params}: {e}")
        stiffness_props = {v: None for v in CROSS_SECTION_PARAMETERS}
        valid = False

    result = Sample(
        section_type=input.section_type,
        section_params=input.section_params,
        valid=valid,
        stiffness_props=stiffness_props,
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
        "--num_sections", type=int, default=-1, help="Number of sections to generate"
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

        if (num_sections := args.num_sections) < 0:
            num_sections = config["num_sections"]
        assert num_sections > 0, "Number of sections must be positive."

        batch_size = args.batch_size
        num_workers = args.num_workers
        output = args.output

        logger.info(f"Generating {num_sections} samples in batches of {batch_size}...")
        total_batches = (num_sections + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_sections)
            batch_count = end_idx - start_idx
            logger.info(f"Processing batch {batch_idx+1}/{total_batches} ({batch_count} samples)...")

            tasks = []
            for _ in range(batch_count):
                section_params = random_section_params(section_data)
                mesh_sizes = section_data["mesh_sizes"]
                sample_input = SampleInput(
                    section_type=section_type,
                    section_params=section_params,
                    mesh_sizes=mesh_sizes,
                    material_params=material_params,
                )
                tasks.append(sample_input)

            data = []
            with multiprocessing.Pool(num_workers) as pool:
                with tqdm(total=len(tasks), desc=f"Batch {batch_idx+1}/{total_batches}") as pbar:
                    def on_success(result: Any) -> None:
                        data.append(result)
                        pbar.update(1)
                    def on_error(e: Exception) -> None:
                        logger.error(f"Error in worker: {e}")
                        pbar.update(1)
                    for t in tasks:
                        pool.apply_async(
                            generate_sample,
                            args=(t,),
                            callback=on_success,
                            error_callback=on_error,
                        )
                    pool.close()
                    pool.join()

            df = pd.DataFrame(data)
            # Write header only for first batch
            write_header = batch_idx == 0
            df.to_csv(output, mode='a', header=write_header, index=False)
            logger.info(f"Appended batch {batch_idx+1} to {output}.")
        logger.info(f"Data generation completed. All batches saved to {output}.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(generate_learning_data())
