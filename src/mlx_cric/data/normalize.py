"""Team + venue normalization shared by all loaders and live inference."""
from __future__ import annotations

import re

# CricAPI "India [IND]" / "Sri Lanka Women" / legacy variants -> canonical
TEAM_ALIASES = {
    "india [ind]": "India",
    "west indies [wi]": "West Indies",
    "australia [aus]": "Australia",
    "england [eng]": "England",
    "pakistan [pak]": "Pakistan",
    "south africa [rsa]": "South Africa",
    "south africa [sa]": "South Africa",
    "new zealand [nz]": "New Zealand",
    "sri lanka [sl]": "Sri Lanka",
    "bangladesh [ban]": "Bangladesh",
    "afghanistan [afg]": "Afghanistan",
    "zimbabwe [zim]": "Zimbabwe",
    "ireland [ire]": "Ireland",
    "mumbai indians [mi]": "Mumbai Indians",
    "chennai super kings [csk]": "Chennai Super Kings",
    "royal challengers bangalore [rcb]": "Royal Challengers Bangalore",
    "royal challengers bengaluru": "Royal Challengers Bangalore",
    "kolkata knight riders [kkr]": "Kolkata Knight Riders",
    "delhi capitals": "Delhi Daredevils",
    "punjab kings": "Kings XI Punjab",
    "sunrisers hyderabad [srh]": "Sunrisers Hyderabad",
    "rajasthan royals [rr]": "Rajasthan Royals",
    # Short forms seen in the doosra corpus (frequency-checked):
    # KKR/RCB are unambiguous IPL abbreviations; bare "Kings XI",
    # "Super Kings" and "Sunrisers" are dominated by their IPL sides.
    "kkr": "Kolkata Knight Riders",
    "rcb": "Royal Challengers Bangalore",
    "kings xi": "Kings XI Punjab",
    "super kings": "Chennai Super Kings",
    "sunrisers": "Sunrisers Hyderabad",
}


def normalize_team(s: str) -> str:
    t = str(s or "").strip()
    t = re.sub(r"\s*\[.*?\]\s*$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    low = t.lower()
    if low in TEAM_ALIASES:
        return TEAM_ALIASES[low]
    # keep Women suffix canonical: "India Women" not "India women"
    if low.endswith(" women"):
        base = normalize_team(t[: -len(" women")])
        return f"{base} Women"
    return t


# Major cricket cities -> country (covers ~95% of doosra venues; rest -> Unknown)
CITY_TO_COUNTRY = {
    "Mumbai": "India", "Bengaluru": "India", "Bangalore": "India", "Chennai": "India",
    "Kolkata": "India", "Delhi": "India", "New Delhi": "India", "Hyderabad": "India",
    "Ahmedabad": "India", "Pune": "India", "Jaipur": "India", "Mohali": "India",
    "Chandigarh": "India", "Nagpur": "India", "Kanpur": "India", "Lucknow": "India",
    "Ranchi": "India", "Indore": "India", "Rajkot": "India", "Vadodara": "India",
    "Dharamsala": "India", "Guwahati": "India", "Thiruvananthapuram": "India",
    "Visakhapatnam": "India", "Cuttack": "India", "Sydney": "Australia",
    "Melbourne": "Australia", "Brisbane": "Australia", "Perth": "Australia",
    "Adelaide": "Australia", "Hobart": "Australia", "Canberra": "Australia",
    "Darwin": "Australia", "Geelong": "Australia",
    "London": "England", "Birmingham": "England", "Manchester": "England",
    "Leeds": "England", "Nottingham": "England", "Southampton": "England",
    "Cardiff": "England", "Chester-le-Street": "England", "Bristol": "England",
    "Taunton": "England", "Derby": "England",
    # English county grounds (domestic + internationals hosted there)
    "Canterbury": "England", "Chelmsford": "England", "Hove": "England",
    "Northampton": "England", "Worcester": "England", "Leicester": "England",
    "Durham": "England", "Swansea": "England", "Southend-on-Sea": "England",
    "Milton Keynes": "England", "Solihull": "England", "Coggeshall": "England",
    "Karachi": "Pakistan", "Lahore": "Pakistan", "Rawalpindi": "Pakistan",
    "Multan": "Pakistan", "Faisalabad": "Pakistan", "Peshawar": "Pakistan",
    "Sharjah": "United Arab Emirates", "Dubai": "United Arab Emirates",
    "Abu Dhabi": "United Arab Emirates",
    "Cape Town": "South Africa", "Johannesburg": "South Africa",
    "Durban": "South Africa", "Centurion": "South Africa",
    "Gqeberha": "South Africa", "Port Elizabeth": "South Africa",
    "Bloemfontein": "South Africa", "Paarl": "South Africa",
    "Auckland": "New Zealand", "Wellington": "New Zealand",
    "Christchurch": "New Zealand", "Hamilton": "New Zealand",
    "Dunedin": "New Zealand", "Napier": "New Zealand", "Nelson": "New Zealand",
    "Colombo": "Sri Lanka", "Kandy": "Sri Lanka", "Galle": "Sri Lanka",
    "Dambulla": "Sri Lanka", "Hambantota": "Sri Lanka", "Pallekele": "Sri Lanka",
    "Bridgetown": "West Indies", "Kingston": "West Indies",
    "Port of Spain": "West Indies", "Georgetown": "West Indies",
    "St Lucia": "West Indies", "Gros Islet": "West Indies",
    "North Sound": "West Indies", "St Kitts": "West Indies", "Basseterre": "West Indies",
    "Providence": "West Indies", "Roseau": "West Indies", "Tarouba": "West Indies",
    "Dhaka": "Bangladesh", "Mirpur": "Bangladesh", "Chattogram": "Bangladesh",
    "Chittagong": "Bangladesh", "Sylhet": "Bangladesh", "Fatullah": "Bangladesh",
    "Khulna": "Bangladesh",
    "Harare": "Zimbabwe", "Bulawayo": "Zimbabwe",
    "Kabul": "Afghanistan",
    "Dublin": "Ireland", "Belfast": "Ireland", "Bready": "Ireland", "Malahide": "Ireland",
    "Amsterdam": "Netherlands", "Rotterdam": "Netherlands", "The Hague": "Netherlands",
    "Edinburgh": "Scotland", "Glasgow": "Scotland", "Aberdeen": "Scotland",
    "Windhoek": "Namibia", "Kirtipur": "Nepal", "Kathmandu": "Nepal",
    "Al Amerat": "Oman", "Muscat": "Oman",
    "Port Moresby": "Papua New Guinea",
    "Lauderhill": "USA", "Dallas": "USA", "Grand Prairie": "USA", "New York": "USA",
    "Toronto": "Canada", "King City": "Canada", "Brampton": "Canada",
    "Nairobi": "Kenya",
    "Hong Kong": "Hong Kong", "Kowloon": "Hong Kong",
    "Singapore": "Singapore", "Kuala Lumpur": "Malaysia",
    "Krefeld": "Germany", "Berlin": "Germany",
    "Europa Point": "Gibraltar", "Marsa": "Israel", "Douglas": "Isle of Man",
}


def infer_country(city: str, stadium: str = "") -> str:
    c = str(city or "").strip()
    if c in CITY_TO_COUNTRY:
        return CITY_TO_COUNTRY[c]
    # stadium often contains city: "Melbourne Cricket Ground"
    for known_city, country in CITY_TO_COUNTRY.items():
        if known_city.lower() in c.lower() or known_city.lower() in str(stadium).lower():
            return country
    return "Unknown"
