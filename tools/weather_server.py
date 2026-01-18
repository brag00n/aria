from mcp.server.fastmcp import FastMCP

# Nom du serveur (doit correspondre à la conf default.json si utilisé en clé, 
# mais ici c'est surtout pour les logs internes)
mcp = FastMCP("meteo_local")

def fetch_weather_logic(city: str):
    """Logique isolée."""
    city_clean = city.lower().strip()
    data = {
        "paris": "Pluvieux, 12°C",
        "lyon": "Ensoleillé, 18°C",
        "marseille": "Mistral fort, 22°C",
        "bordeaux": "Nuageux, 15°C",
        "toulouse": "Vent d'autan, 19°C"
    }
    # Retourne une phrase complète pour qu'Héléna (TTS) la lise naturellement
    return data.get(city_clean, f"Je n'ai pas d'information météo pour {city}.")

@mcp.tool()
def get_weather(city: str):
    """
    Donne la météo pour une ville.
    
    Args:
        city: Le nom de la ville (ex: Paris, Lyon).
    """
    # Pas de "-> str" ici pour éviter le crash Pydantic v2
    return fetch_weather_logic(city)

if __name__ == "__main__":
    mcp.run()