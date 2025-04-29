from pydantic import BaseModel

class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Deck: str

from pydantic import BaseModel, Field

class Passenger(BaseModel):
    Pclass: int = Field(..., gt=0, lt=4, description="Ticket class (1, 2, or 3)")
    Sex: str = Field(..., pattern="^(male|female)$")
    Age: float = Field(..., gt=0, lt=100)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., gt=0)
    Embarked: str = Field(..., pattern="^(S|C|Q)$")
    Deck: str = Field(..., min_length=1, max_length=1)