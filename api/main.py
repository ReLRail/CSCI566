from fastapi import FastAPI
import keyboard

app = FastAPI()


@app.get("/spell")
async def cast_spell(spell_id):
    keyboard.write(f'{spell_id}',delay=0)
    return f"key [{spell_id}] pressed!"