class OpisZestawu:
    def __init__(self, json_data: dict):
        self.meta: dict = json_data.get("meta", {})
        self.variables: dict[str, dict[str, str | list[str]]] = json_data.get("variables", {})

        # Graf ekspercki, może nie zawierać niektórych prawdziwych zależności
        # i może zawierać zależności, które nie istnieją w rzeczywistości.
        self.expert_edges: list[list[str]] = json_data.get("expert_edges", [])
        
        # Krawędzie, które napewno są obecne 
        self.required_edges: list[list[str]] = json_data.get("required_edges", [])

        # Krawędzie, które napewno nie ma
        self.forbidden_edges: list[list[str]] = json_data.get("forbidden_edges", [])

        # To warstwy czasowe, zmienna z późniejszej warstwy nie może wpływać na zmienną z wcześniejszej warstwy
        self.tiers: list[list[str]] = json_data.get("tiers", [])

        