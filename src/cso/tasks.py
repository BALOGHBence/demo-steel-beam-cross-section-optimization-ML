from enum import Enum


class Task(Enum):
    FAILURE_PREDICTION = "failure_prediction"
    UTILIZATION_ESTIMATION = "utilization_estimation"
    SECTION_ESTIMATION = "section_estimation"
    AREA_ESTIMATION = "area_estimation"
    GEOMETRY_VALIDATION = "geometry_validation"
    

if __name__ == "__main__":
    for task in Task:
        print(f"Defined task: {task.value}")