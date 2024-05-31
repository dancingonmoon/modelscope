from typing import Union, Annotated
from fastapi import FastAPI, Path
from pydantic import BaseModel

description = """
## 一个例子,🦬
 + **create user**
 + **get user**
"""
app = FastAPI( title="FastAPI Example", description=description,)

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float , None] = None
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(user_id:str, item_id:str, q: Union[str, None] = None, short: bool = False):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item["q"] = q
    if not short:
        item['description'] = "long description"
    return item

@app.put("/items/{item_id}")
async def update_item(
    item_id: Annotated[int, Path(title="The ID of the item to get", ge=0, le=1000)],
    q: Union[str, None] = None,
    item: Union[Item, None] = None,
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    if item:
        results.update({"item": item})
    return results

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080, )
