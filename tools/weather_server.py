# weather_server.py
from mcp.server.fastmcp import FastMCP

# Initialisation du serveur "FastMCP" (haute performance)
mcp = FastMCP("MeteoLocale")

@mcp.tool()
def get_weather(city: str) -> str:
    """Donne la météo actuelle pour une ville donnée."""
    # Simulation de données (ici on ferait un appel API réel)
    data = {
        "paris": "Pluvieux, 12°C",
        "lyon": "Ensoleillé, 18°C",
        "marseille": "Mistral fort, 22°C"
    }
    return data.get(city.lower(), "Météo inconnue pour cette ville.")

if __name__ == "__main__":
    # Lance le serveur sur stdio (entrée/sortie standard)
    mcp.run()