from pydantic import BaseModel, Field

class OptimizeRequest(BaseModel):
    """
    Request body for the optimization endpoint.

    """

    process_duration_hours: int = Field(
        default=4,
        ge=1,
        description="Duration of the process in hours."
    )

    earliest_start: int = Field(
        default=6,
        ge=0,
        le=23,
        description="Earliest allowed start hour."
    )

    latest_end: int= Field(
        default=22,
        ge=1,
        le=24,
        description="Latest allowed end hour."        
    )

    baseline_start: int = Field(
        default=8,
        ge=0,
        le=23,
        description="Baseline process starat hour."
    )

    baseline_end: int = Field(
        default=12,
        ge=1,
        le=24,
        description="Baseline process end hour."
    )

    class OptimizeResponse(BaseModel):
        """
        Response returned by the optimization endpoint.
        """
        recommended_start: str
        recommended_end: str

        estimated_cost: float
        baseline_cost: float

        saving: float
        saving_percent: float


        