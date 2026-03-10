from __future__ import annotations

REAL_PLAYERS: list[tuple[str, str]] = [
    ("Mudda", "Australia"),
    ("Carl Jr.", "Canada"),
    ("Massa", "Germany"),
    ("Binkss", "France"),
    ("Wosile", "France"),
    ("GranaDy", "Germany"),
    ("Bren", "France"),
    ("Nayko", "France"),
    ("Epos", "United Kingdom"),
    ("Otaaaq", "France"),
    ("Gwen", "France"),
    ("Xerar", "France"),
    ("Scrapie", "Belgium"),
    ("eLconn21", "Czechia"),
    ("Tona", "France"),
    ("Heav", "France"),
    ("Legu", "France"),
    ("Whizzy", "United Kingdom"),
    ("josh1248", "United Kingdom"),
    ("jdon", "United Kingdom"),
    ("pusztitopako", "Hungary"),
    ("Pac", "United Kingdom"),
    ("Soulja", "Belgium"),
    ("Cocow", "France"),
    ("Stufts", "Netherlands"),
    ("Melioo", "France"),
    ("Azion", "France"),
    ("NiTech", "Russia"),
    ("RotakeR", "Poland"),
    ("Affi", "Switzerland"),
    ("Glast", "France"),
    ("Snow", "France"),
    ("Hyll", "Czechia"),
    ("Letzter", "Germany"),
    ("NuPrime", "Germany"),
    ("Charles", "Colombia"),
    ("Razii", "Croatia"),
    ("mime", "Poland"),
    ("Flimsy", "United States"),
    ("neon", "Poland"),
    ("Jan123405", "Slovenia"),
    ("Worker", "France"),
    ("Molle", "Sweden"),
    ("Spammiej", "Netherlands"),
    ("Eria", "Belgium"),
    ("tween", "Slovakia"),
    ("Tjalic", "France"),
    ("J_Swag", "New Zealand"),
    ("Birdie", "Turkey"),
    ("Bimy", "Russia"),
    ("Cemko", "Turkey"),
    ("L1ngo", "Canada"),
    ("Jan", "Germany"),
    ("Evon", "Poland"),
    ("Kakne", "United States"),
    ("Tricky", "France"),
    ("Yekcosdo", "Portugal"),
    ("bestie77", "Czechia"),
    ("Dionysos", "France"),
    ("noiszia", "Argentina"),
    ("ender", "Morocco"),
    ("Dog", "Germany"),
    ("Cinxsss", "Norway"),
    ("MimoJr", "France"),
    ("Feed", "France"),
    ("Neal", "United States"),
    ("Goldennugg3t", "Germany"),
    ("Magorian", "Poland"),
    ("Mobbi", "Germany"),
    ("Aurel", "France"),
    ("Huso", "Turkey"),
    ("Richie1308", "France"),
    ("key", "Poland"),
    ("Tomczan", "Poland"),
    ("Haedra", "France"),
    ("Kurisu-tina", "Italy"),
    ("raizo", "Slovakia"),
    ("link", "France"),
    ("PauLL", "Portugal"),
    ("Pika", "Portugal"),
    ("AwayFridish", "Italy"),
    ("Panda", "France"),
    ("purple", "Germany"),
    ("Ricso5", "Hungary"),
    ("baiwack", "New Zealand"),
    ("Barbos", "Germany"),
    ("oNio", "Germany"),
    ("TaMaR", "Israel"),
    ("Narcor", "Poland"),
    ("Brinkenn", "Sweden"),
    ("Liquid", "United States"),
    ("Ener", "France"),
    ("Chamow", "France"),
    ("Javzo", "Netherlands"),
    ("Intax", "Norway"),
    ("Spark", "India"),
    ("Demotivator13", "Poland"),
    ("V1NCH", "France"),
]


def parse_standings(text: str) -> list[tuple[str, str]]:
    """Parse raw standings text into [(name, country), ...].

    Expects lines like: '1. PlayerName CountryName' or 'PlayerName CountryName'.
    Handles multi-word countries: 'United Kingdom', 'United States', 'New Zealand'.
    """
    multi_word_countries = {
        "United Kingdom",
        "United States",
        "New Zealand",
    }
    results: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip optional leading rank number
        if line[0].isdigit():
            parts = line.split(None, 1)
            if len(parts) == 2:
                line = parts[1].lstrip(". ")
        tokens = line.split()
        if len(tokens) < 2:
            continue
        # Try two-word country first
        for country_words in (2, 1):
            if len(tokens) > country_words:
                name = " ".join(tokens[:-country_words])
                country = " ".join(tokens[-country_words:])
                if country_words == 1 or country in multi_word_countries:
                    results.append((name, country))
                    break
    return results
