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

from cso.section import construct_section, section_properties, random_section_params
from cso.logger import get_logger, set_log_level
from cso.constants import CROSS_SECTION_PARAMETERS

load_dotenv()
logger = get_logger()
set_log_level("INFO")


def generate_sample(args: tuple[dict, dict]) -> dict:
    """Generate a single data sample of section parameters, loads, and utilization."""
    (
        section_type,
        section_params,
        mesh_sizes,
        material_params,
    ) = args
    try:
        section = construct_section(
            geometry_constructor=section_type,
            params=section_params,
            material=material_params,
            mesh_sizes=mesh_sizes,
            calculate=True,
        )
        stiffness_props = section_properties(section)
        valid = True
    except Exception as e:
        if "TopologyException" in str(e):
            logger.debug(f"TopologyException for params {section_params}: {e}")
        else:
            logger.error(f"Error generating sample for params {section_params}: {e}")
        stiffness_props = {v: None for v in CROSS_SECTION_PARAMETERS}
        valid = False

    result = {
        **section_params,
        **stiffness_props,
        "section_type": section_type,
        "valid": valid,
    }
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

        logger.info("Generating random section parameters...")
        logger.debug(f"Number of sections: {num_sections}")
        tasks = []
        for _ in range(num_sections):
            section_params = random_section_params(section_data)
            mesh_sizes = section_data["mesh_sizes"]
            tasks.append((section_type, section_params, mesh_sizes, material_params))

        logger.info(f"Generated {len(tasks)} tasks.")

        num_workers = args.num_workers
        logger.info(f"Generating data with {num_workers} parallel workers...")

        data = []
        with multiprocessing.Pool(num_workers) as pool:
            with tqdm(total=len(tasks), desc="Generating samples") as pbar:

                def on_success(result: Any) -> None:
                    """Called after each successful worker execution."""
                    data.append(result)
                    pbar.update(1)

                def on_error(e: Exception) -> None:
                    """Called after a failed worker execution."""
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

        logger.info(f"Data generation completed. Saving results to {args.output} ...")
        df = pd.DataFrame(data)
        df.to_csv(args.output, index=False)
        logger.info(f"Saved results to {args.output}.")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(generate_learning_data())
