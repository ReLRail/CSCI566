from fastapi import FastAPI
import pyautogui

app = FastAPI()


@app.get("/spell")
async def read_item(spell_id):
    pyautogui.press(f"{spell_id}")
    return f"key [{spell_id}] pressed!"