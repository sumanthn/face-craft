"""
Diverse celebrity list for bias testing - 250+ celebrities across regions, ages, and feature types
"""

CELEBRITIES = {
    # EAST ASIA (40)
    "east_asia": [
        # China
        "Fan Bingbing", "Gong Li", "Zhang Ziyi", "Liu Yifei", "Li Bingbing",
        "Huang Xiaoming", "Wang Leehom", "Jay Chou", "Yang Mi", "Zhao Wei",
        "Tang Wei", "Zhou Xun", "Chen Kun", "Deng Chao", "Sun Li",
        # Japan
        "Ken Watanabe", "Rinko Kikuchi", "Takeshi Kaneshiro", "Hiroyuki Sanada", "Tao Okamoto",
        "Kiko Mizuhara", "Haruma Miura", "Satoshi Tsumabuki", "Meisa Kuroki", "Nanao",
        # Korea
        "Song Hye-kyo", "Bae Doona", "Lee Byung-hun", "Gong Yoo", "Jun Ji-hyun",
        "Kim Soo-hyun", "Park Shin-hye", "Lee Min-ho", "Song Joong-ki", "IU",
        "Jennie Kim", "V BTS", "Lisa Blackpink", "Hyun Bin", "Son Ye-jin",
    ],

    # SOUTH ASIA (35)
    "south_asia": [
        # India
        "Aishwarya Rai", "Priyanka Chopra", "Deepika Padukone", "Hrithik Roshan", "Shah Rukh Khan",
        "Aamir Khan", "Kajol", "Kareena Kapoor", "Ranbir Kapoor", "Alia Bhatt",
        "Ranveer Singh", "Anushka Sharma", "Katrina Kaif", "Vidya Balan", "Tabu",
        "Amitabh Bachchan", "Irrfan Khan", "Nawazuddin Siddiqui", "Sonam Kapoor", "Shraddha Kapoor",
        "Disha Patani", "Varun Dhawan", "Siddharth Malhotra", "Kiara Advani", "Vicky Kaushal",
        # Pakistan
        "Mahira Khan", "Fawad Khan", "Atif Aslam", "Ayeza Khan", "Sajal Aly",
        # Bangladesh/Sri Lanka
        "Jacqueline Fernandez", "Pooja Hegde", "Nayanthara", "Samantha Ruth Prabhu", "Vijay Deverakonda",
    ],

    # SOUTHEAST ASIA (25)
    "southeast_asia": [
        # Various
        "Michelle Yeoh", "Tony Leung", "Maggie Cheung", "Chow Yun-fat", "Andy Lau",
        "Donnie Yen", "Jet Li", "Ziyi Zhang", "Kelly Chen", "Shu Qi",
        # Thailand
        "Mario Maurer", "Davika Hoorne", "Urassaya Sperbund", "Nadech Kugimiya",
        # Philippines
        "Nadine Lustre", "James Reid", "Liza Soberano", "Enrique Gil", "Anne Curtis",
        # Indonesia
        "Raline Shah", "Joe Taslim", "Chelsea Islan", "Iko Uwais",
        # Vietnam
        "Tang Thanh Ha", "Ngo Thanh Van",
    ],

    # MIDDLE EAST & NORTH AFRICA (30)
    "middle_east_north_africa": [
        # Iran/Persia
        "Golshifteh Farahani", "Nazanin Boniadi", "Shohreh Aghdashloo", "Sarah Shahi",
        # Arab World
        "Omar Sharif", "Haifa Wehbe", "Nancy Ajram", "Elissa", "Myriam Fares",
        "Amr Diab", "Tamer Hosny", "Mohamed Ramadan", "Yasmine Sabri", "Hend Sabry",
        # Turkey
        "Kıvanç Tatlıtuğ", "Beren Saat", "Çağatay Ulusoy", "Fahriye Evcen", "Burak Özçivit",
        "Hande Erçel", "Kerem Bürsin", "Tuba Büyüküstün", "Engin Akyürek",
        # Israel
        "Gal Gadot", "Natalie Portman", "Bar Refaeli", "Odeya Rush",
        # Egypt
        "Rami Malek", "Mena Massoud", "Ahmed Zaki",
    ],

    # SUB-SAHARAN AFRICA (30)
    "sub_saharan_africa": [
        # Nigeria
        "Genevieve Nnaji", "Omotola Jalade", "Ramsey Nouah", "Chiwetel Ejiofor", "John Boyega",
        "David Oyelowo", "Cynthia Erivo", "Uzo Aduba", "Adewale Akinnuoye-Agbaje",
        # Kenya/East Africa
        "Lupita Nyong'o", "Wanuri Kahiu", "Edi Gathegi",
        # South Africa
        "Charlize Theron", "Trevor Noah", "Sello Maake Ka-Ncube", "Sharlto Copley", "Nomzamo Mbatha",
        "Pearl Thusi", "Bonang Matheba",
        # West Africa
        "Idris Elba", "Djimon Hounsou", "Akon", "Seal", "Naomi Campbell",
        # Other
        "Danai Gurira", "Letitia Wright", "Daniel Kaluuya", "Thandiwe Newton", "Sophie Okonedo",
    ],

    # LATIN AMERICA (35)
    "latin_america": [
        # Mexico
        "Salma Hayek", "Gael García Bernal", "Diego Luna", "Eiza González", "Yalitza Aparicio",
        "Eugenio Derbez", "Kate del Castillo", "Demián Bichir", "Adriana Lima",
        # Brazil
        "Gisele Bündchen", "Morena Baccarin", "Alice Braga", "Rodrigo Santoro", "Sonia Braga",
        "Wagner Moura", "Fernanda Montenegro", "Camila Pitanga", "Taís Araújo",
        # Colombia
        "Sofía Vergara", "Shakira", "John Leguizamo", "Maluma", "J Balvin",
        # Argentina
        "Bérénice Bejo", "Ricardo Darín", "Cecilia Roth",
        # Cuba/Caribbean
        "Ana de Armas", "Oscar Isaac", "Rosario Dawson", "Zoe Saldana", "Cameron Diaz",
        # Other
        "Pedro Pascal", "Stephanie Sigman", "Aubrey Plaza", "America Ferrera",
    ],

    # INDIGENOUS & PACIFIC ISLANDER (15)
    "indigenous_pacific": [
        # Maori/Pacific
        "Taika Waititi", "Jason Momoa", "Cliff Curtis", "Temuera Morrison", "Keisha Castle-Hughes",
        "Jemaine Clement", "Rachel House",
        # Native American/First Nations
        "Devery Jacobs", "Wes Studi", "Adam Beach", "Graham Greene", "Irene Bedard",
        "Martin Sensmeier", "Zahn McClarnon", "Tantoo Cardinal",
    ],

    # EUROPE - WESTERN (30)
    "europe_western": [
        # France
        "Léa Seydoux", "Marion Cotillard", "Vincent Cassel", "Omar Sy", "Eva Green",
        "Audrey Tautou", "Jean Dujardin", "Mélanie Laurent",
        # UK
        "Idris Elba", "Emily Blunt", "Tom Hiddleston", "Keira Knightley", "Dev Patel",
        "Benedict Cumberbatch", "Emma Watson", "Daniel Craig", "Judi Dench", "Helen Mirren",
        # Spain
        "Penélope Cruz", "Javier Bardem", "Antonio Banderas", "Úrsula Corberó", "Álvaro Morte",
        # Italy
        "Monica Bellucci", "Sophia Loren", "Luca Marinelli",
        # Germany
        "Diane Kruger", "Daniel Brühl", "Nina Hoss",
    ],

    # EUROPE - NORTHERN (20)
    "europe_northern": [
        # Scandinavia
        "Mads Mikkelsen", "Alicia Vikander", "Alexander Skarsgård", "Noomi Rapace",
        "Stellan Skarsgård", "Rebecca Ferguson", "Joel Kinnaman", "Nikolaj Coster-Waldau",
        "Brigitte Nielsen", "Dolph Lundgren", "Ingrid Bergman legacy",
        # Netherlands/Belgium
        "Carice van Houten", "Famke Janssen", "Matthias Schoenaerts",
        # Ireland
        "Saoirse Ronan", "Cillian Murphy", "Colin Farrell", "Ruth Negga", "Barry Keoghan",
        "Paul Mescal",
    ],

    # EUROPE - EASTERN (20)
    "europe_eastern": [
        # Russia
        "Milla Jovovich", "Natalia Vodianova", "Irina Shayk", "Olga Kurylenko", "Svetlana Khodchenkova",
        # Poland
        "Joanna Kulig", "Robert Więckiewicz",
        # Czech/Slovak
        "Petra Němcová", "Karolína Kurková",
        # Romania/Hungary
        "Sebastian Stan", "Kate Beckinsale",
        # Ukraine
        "Mila Kunis", "Olga Kurylenko", "Vera Farmiga",
        # Serbia/Balkans
        "Rade Šerbedžija", "Goran Višnjić",
        # Other
        "Kristin Kreuk", "Nina Dobrev", "Emmanuelle Chriqui",
    ],

    # AUSTRALIA/NEW ZEALAND (15)
    "oceania": [
        "Cate Blanchett", "Nicole Kidman", "Hugh Jackman", "Chris Hemsworth", "Margot Robbie",
        "Heath Ledger legacy", "Russell Crowe", "Naomi Watts", "Rose Byrne", "Toni Collette",
        "Sam Neill", "Anna Torv", "Yvonne Strahovski", "Rebel Wilson", "Melissa George",
    ],

    # MIXED HERITAGE / MULTIRACIAL (20)
    "mixed_heritage": [
        "Keanu Reeves", "Halle Berry", "Rashida Jones", "Meghan Markle", "Trevor Noah",
        "Vin Diesel", "Dwayne Johnson", "Barack Obama", "Tiger Woods", "Lewis Hamilton",
        "Zendaya", "Yara Shahidi", "Amandla Stenberg", "Lenny Kravitz", "Tracee Ellis Ross",
        "Thandiwe Newton", "Maya Rudolph", "Wentworth Miller", "Jesse Williams", "Zazie Beetz",
    ],

    # AGE DIVERSITY - 50+ (15)
    "age_50_plus": [
        "Meryl Streep", "Helen Mirren", "Judi Dench", "Viola Davis", "Angela Bassett",
        "Denzel Washington", "Morgan Freeman", "Samuel L Jackson", "George Clooney", "Brad Pitt",
        "Michelle Pfeiffer", "Glenn Close", "Julianne Moore", "Frances McDormand", "Annette Bening",
    ],
}

# Flatten for easy iteration
def get_all_celebrities():
    all_celebs = []
    for region, celebs in CELEBRITIES.items():
        for celeb in celebs:
            all_celebs.append({"name": celeb, "region": region})
    return all_celebs

if __name__ == "__main__":
    all_celebs = get_all_celebrities()
    print(f"Total celebrities: {len(all_celebs)}")
    for region, celebs in CELEBRITIES.items():
        print(f"  {region}: {len(celebs)}")
