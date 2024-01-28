# from typing import Union
from fastapi import FastAPI
from service.api.api import main_router


# from pydantic import BaseModel

# app = FastAPI()
# class InputItem(BaseModel):
#     name: str
#     price: int
#     discount: int

# class OutputItem(BaseModel):
#     name: str
#     selling_price: int
#
# # get, post, delete
# @app.get("/")
# def read():
#     return {"Hello": "World"}
#
#
# @app.post("/items/", response_model = OutputItem)
# def add_item(item: InputItem):
#     selling_price = item.price - item.discount
#     return {" name": item.name, "selling_price": selling_price}

app = FastAPI(project_name="Movie Recommendation")
app.include_router(main_router)

@app.get("/")
def root():
    return {"hello": "world"}



