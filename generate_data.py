import csv
import os
import sys
import shutil
import tempfile
import json
from glob import glob
from multiprocessing import Process, Queue
from queue import Empty
from tqdm import tqdm
from dotenv import load_dotenv
import argparse
from argparse import Namespace
from pydantic import BaseModel
from sectionproperties.analysis import Section
from scipy.optimize import minimize_scalar, OptimizeResult

from cso.section import (
    construct_section,
    utilization as calculate_utilization,
    section_properties,
    find_internal_force_limits,
    default_section_params,
    random_section_params,
)
from cso.loads import random_loads
from cso.material import MaterialProperties, random_material_params
from cso.constants import CROSS_SECTION_PARAMETERS
from cso.logger import get_logger, set_log_level

load_dotenv()
logger = get_logger()
set_log_level("INFO")

SENTINEL = "STOP"  # stop signal for workers
MSG_DATA = "DATA"  # output_q message type: data record
MSG_DONE = "DONE"  # output_q message type: worker finished


class Sample(BaseModel):
    """Data model representing a generated sample."""

    section_type: str
    section_params: dict
    valid: bool
    stiffness_props: dict
    material_props: dict
    load_params: dict
    utilization: float

    def as_flat_dict(self) -> dict:
        """Returns a flattened dictionary representation of the sample."""
        flat_dict = {
            "section_type": self.section_type,
            "valid": self.valid,
        }
        flat_dict.update({k: v for k, v in self.section_params.items()})
        flat_dict.update({k: v for k, v in self.stiffness_props.items()})
        flat_dict.update({k: v for k, v in self.material_props.items()})
        flat_dict.update({k: v for k, v in self.load_params.items()})
        flat_dict["utilization"] = self.utilization
        return flat_dict
    
    
def find_load_multiplier(section: Section, load_params: dict, target_utilization: float, tol=0.1) -> dict:
    """Finds a load multiplier for a target utilization within a tolerance."""
    def objective(scale: float) -> float:
        scaled_loads = {k: v * scale for k, v in load_params.items()}
        utilization_value = calculate_utilization(section, scaled_loads)
        return abs(utilization_value - target_utilization)
    
    res: OptimizeResult = minimize_scalar(
        objective,
        method='brent',
        options={'xtol': tol}
    )
    
    if res.success:
        return res.x
    else:
        raise ValueError("Load scaling optimization failed.")
    
    
def worker(
    worker_id: int,
    input_q: Queue,
    output_q: Queue,
    section_type: str,
    section_data: dict,
    material_data: dict,
    load_ranges: dict,
) -> None:
    """
    Worker:
    - Continuously generates records
    - Periodically checks the input queue for a SENTINEL
    - Sends records to the output queue
    - At shutdown, sends a 'DONE' message
    """
    while True:
        # Check for stop signal (non-blocking)
        try:
            msg = input_q.get_nowait()
            if msg == SENTINEL:
                break
            # if some other kind of message appears, ignore it
        except Empty:
            pass

        section_params = random_section_params(section_data)
        material_params = random_material_params(material_data)
        mesh_sizes = section_data["mesh_sizes"]
        load_params = random_loads(load_ranges)
        samples = []
        try:
            section = construct_section(
                geometry_constructor=section_type,
                params=section_params,
                material=material_params,
                mesh_sizes=mesh_sizes,
                calculate=True,
            )
            stiffness_props: dict = section_properties(section)
            
            factor = find_load_multiplier(section, load_params, target_utilization=0.9, tol=0.01)
            loads = {k: v * factor for k, v in load_params.items()}
            util = calculate_utilization(section, loads)
            samples.append((stiffness_props, loads, util))
            
            # factor /= 2
            # loads = {k: v * factor for k, v in load_params.items()}
            # util = calculate_utilization(section, loads)
            # samples.append((stiffness_props, loads, util))
            
            factor = find_load_multiplier(section, load_params, target_utilization=1.1, tol=0.01)
            loads = {k: v * factor for k, v in load_params.items()}
            util = calculate_utilization(section, loads)
            samples.append((stiffness_props, loads, util))
            
            # factor *= 2
            # loads = {k: v * factor for k, v in load_params.items()}
            # util = calculate_utilization(section, loads)
            # samples.append((stiffness_props, loads, util))
            
            valid = True
        except Exception:
            util = None
            stiffness_props = {v: None for v in CROSS_SECTION_PARAMETERS}
            samples.append((stiffness_props, load_params, util))
            valid = False

        material_props = MaterialProperties(**material_params)
        relevant_material_props = {
            "elastic_modulus": material_props.elastic_modulus,
            "poissons_ratio": material_props.poissons_ratio,
            "yield_strength": material_props.yield_strength,
        }

        for stiffness_props, load_params, util in samples:
            result = Sample(
                section_type=section_type,
                section_params=section_params,
                valid=valid,
                stiffness_props=stiffness_props,
                material_props=relevant_material_props,
                load_params=load_params,
                utilization=util,
            ).as_flat_dict()

            # Send to output queue
            output_q.put((MSG_DATA, result))

    # Let the main process know this worker is done
    output_q.put((MSG_DONE, worker_id))


def write_chunk_to_csv(
    chunk_records: list[dict], temp_dir: str, chunk_index: int
) -> str:
    """
    Write a batch of records to a new CSV file in temp_dir.
    Returns the path of the created file.
    """
    if not chunk_records:
        return None

    filename = os.path.join(temp_dir, f"chunk_{chunk_index:04d}.csv")
    fieldnames = chunk_records[0].keys()

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chunk_records)

    return filename


def combine_csv_files(temp_dir: str, final_output: str) -> None:
    """
    Combine all chunk_*.csv files in temp_dir into a single CSV file.
    All chunks are assumed to have the same header.
    """
    chunk_files = sorted(glob(os.path.join(temp_dir, "chunk_*.csv")))
    if not chunk_files:
        tqdm.write("No chunk files to combine.")
        return

    tqdm.write(f"Combining {len(chunk_files)} chunk files into {final_output}")

    with open(chunk_files[0], "r", newline="") as f_in:
        reader = csv.reader(f_in)
        header = next(reader)

    with open(final_output, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        for chunk_file in chunk_files:
            with open(chunk_file, "r", newline="") as f_in:
                reader = csv.reader(f_in)
                _ = next(reader)  # skip header in each chunk
                for row in reader:
                    writer.writerow(row)


def calculate_load_ranges(
    section_type: str, section_data: dict, material_params: dict
) -> dict:
    section = construct_section(
        geometry_constructor=section_type,
        params=default_section_params(section_data),
        material=material_params,
        mesh_sizes=section_data["mesh_sizes"],
        calculate=True,
    )
    return find_internal_force_limits(section)


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


def generate_learning_data(args: Namespace) -> None:

    with open(args.config, "r") as f:
        config = json.load(f)

    material_data = config["material"]
    section_data = config["section"]
    section_type = section_data["type"]

    if (num_samples := args.num_samples) < 0:
        num_samples = config["num_samples"]
    assert num_samples > 0, "Number of samples must be positive."

    # Load ranges for random generation
    logger.info("Calculating load ranges based on default section parameters...")
    load_ranges = calculate_load_ranges(section_type, section_data, material_data)
    logger.debug(f"Determined load ranges: {load_ranges}")

    batch_size = args.batch_size
    num_workers = args.num_workers
    output = args.output

    input_q = Queue()
    output_q = Queue(maxsize=batch_size * 2)  # Limits the queue to 2 batches

    # Temporary folder for chunk CSVs
    temp_dir = tempfile.mkdtemp(prefix="data_chunks_")
    tqdm.write(f"Temporary folder: {temp_dir}")

    # Start workers
    workers = []
    for wid in range(num_workers):
        p = Process(
            target=worker,
            args=(
                wid,
                input_q,
                output_q,
                section_type,
                section_data,
                material_data,
                load_ranges,
            ),
            name=f"Worker-{wid}",
        )
        p.start()
        tqdm.write(f"Started {p.name} with PID {p.pid}")
        workers.append(p)

    total_records = 0
    sent_stop_signals = False
    finished_workers = 0
    chunk_index = 0
    buffer = []

    try:
        pbar = tqdm(total=num_samples, desc="Total Records", unit="records")
        while True:
            # If we've generated enough records and haven't yet signaled stop,
            # send a SENTINEL for each worker.
            if not sent_stop_signals and total_records >= num_samples:
                tqdm.write("Target reached, sending stop signals to workers...")
                for _ in range(num_workers):
                    input_q.put(SENTINEL)
                sent_stop_signals = True

            # Try to read from the output queue
            try:
                msg_type, payload = output_q.get(timeout=0.5)
            except Empty:
                # If we've already sent stop signals and all workers are finished,
                # we can break.
                if sent_stop_signals and finished_workers == num_workers:
                    break
                else:
                    continue

            if msg_type == MSG_DATA:
                record = payload
                buffer.append(record)
                total_records += 1
                pbar.update(1)
                pbar.desc = f"Total Records ({total_records}/{num_samples})"

                # If buffer is "full", write to a chunk file
                if len(buffer) >= batch_size:
                    write_chunk_to_csv(buffer, temp_dir, chunk_index)
                    tqdm.write(f"Wrote chunk {chunk_index + 1} with {len(buffer)} records")
                    buffer.clear()
                    chunk_index += 1

            elif msg_type == MSG_DONE:
                finished_workers += 1
                tqdm.write(
                    f"Worker {payload} finished. ({finished_workers}/{num_workers})"
                )
                if sent_stop_signals and finished_workers == num_workers:
                    break

        # After loop: flush remaining buffered records, if any
        if buffer:
            write_chunk_to_csv(buffer, temp_dir, chunk_index)
            tqdm.write(f"Wrote final chunk {chunk_index + 1} with {len(buffer)} records")
            buffer.clear()

    finally:
        # Ensure all workers are joined
        for p in workers:
            p.join()
        pbar.close()

    # Combine chunk CSV files into a single output
    combine_csv_files(temp_dir, output)

    tqdm.write(f"Done. Combined CSV written to {output}")
    tqdm.write(f"Cleaning up temp folder {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    args = parse_args()
    set_log_level(args.loglevel)
    try:
        generate_learning_data(args)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
