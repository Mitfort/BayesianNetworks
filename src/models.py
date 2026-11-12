class OpisZestawu:
    def __init__(self, json_data: dict):
        self.meta: str = json_data.get("meta", "")
        self.vaiables: dict[str, dict[str, str | list[str]]] = {}

        # Graf ekspercki, może nie zawierać niektórych prawdziwych zależności
        # i może zawierać zależności, które nie istnieją w rzeczywistości.
        self.expert_edges: list[list[str]] = []
        
        # Krawędzie, które napewno są obecne 
        self.required_edges: list[list[str]] = []

        # Krawędzie, które napewno nie ma
        self.forbidden_edges: list[list[str]] = []

        # To warstwy czasowe, zmienna z późniejszej warstwy nie może wpływać na zmienną z wcześniejszej warstwy
        self.tiers: list[list[str]] = []

        