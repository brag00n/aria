from mcp.server.fastmcp import FastMCP
import sys

# On garde le nom du serveur cohérent avec default.json
mcp = FastMCP("meteo_local")

def fetch_weather_logic(city: str):
    """Logique isolée pour éviter les conflits d'annotations."""
    city_clean = city.lower().strip()
    data = {
        "paris": "Pluvieux, 12°C",
        "lyon": "Ensoleillé, 18°C",
        "marseille": "Mistral fort, 22°C",
        "bordeaux": "Nuageux, 15°C",
        "toulouse": "Vent d'autan, 19°C"
    }
    return data.get(city_clean, f"Je n'ai pas d'information météo précise pour {city}.")

@mcp.tool(name="get_weather")
def get_weather(city: str):
    """
    Donne la météo pour une ville française.
    
    Args:
        city: Le nom de la ville (ex: Paris, Lyon, Toulouse).
    """
    # Debug vers stderr (ne casse pas le flux JSON-RPC)
    # sys.stderr.write(f"--- APPEL MCP RECU : {city} ---\n")
    
    return fetch_weather_logic(city)

if __name__ == "__main__":
    # Force le mode stdio pour la communication avec Aria
    mcp.run(transport="stdio")