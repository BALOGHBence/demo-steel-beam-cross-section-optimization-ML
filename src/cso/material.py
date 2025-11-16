from pydantic import BaseModel, Field
import numpy as np



class MaterialProperties(BaseModel):
    """
    Data class representing key stiffness properties of a cross section.
    """

    elastic_modulus: float = Field(
        ..., description="Elastic modulus of the material. SI units: [Pa].", gt=0
    )
    poissons_ratio: float = Field(
        ..., description="Poisson's ratio of the material (dimensionless).", gt=-0.5, lt=1.0
    )
    yield_strength: float = Field(
        ..., description="Yield strength of the material. SI units: [Pa].", gt=0
    )
    density: float | None = Field(
        None, description="Density of the material. SI units: [kg/m^3].", gt=0
    )
    color: str | None = Field(
        "gray", description="Color of the material for visualization purposes.", example="red"
    )
    name: str | None = Field(
        "material", description="Name of the material.", example="Structural Steel"
    )
    

def random_material_params(material_data: dict, std: float = 0.1) -> dict:
    """Generate random material parameters, randomized by std.
    
    Parameters
    ----------
    material_data : dict
        Dictionary containing the base material properties.
    std : float, optional
        Standard deviation for the normal distribution used for randomization (default is 0.1).
    """
    params = material_data.copy()
    elastic_modulus = params.get("elastic_modulus")
    poissons_ratio = params.get("poissons_ratio")
    yield_strength = params.get("yield_strength")

    # Randomize each property by normal distribution (mean=original, std=std*original)
    elastic_modulus = float(np.random.normal(elastic_modulus, std * elastic_modulus))
    poissons_ratio = float(np.random.normal(poissons_ratio, std * poissons_ratio))
    yield_strength = float(np.random.normal(yield_strength, std * yield_strength))
    
    # Ensure physical bounds
    poissons_ratio = max(-0.49, min(poissons_ratio, 0.99))
    
    # Ensure reasonable bounds for yield strength and elastic modulus
    yield_strength = max(0.001, yield_strength)
    elastic_modulus = max(0.001, elastic_modulus)

    params.update({
        "elastic_modulus": elastic_modulus,
        "poissons_ratio": poissons_ratio,
        "yield_strength": yield_strength,
    })
    return params