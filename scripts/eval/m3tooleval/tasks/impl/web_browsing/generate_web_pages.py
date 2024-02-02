import os
import json
import random
from faker import Faker

PAGES = [
    # Home page
    dict(
        name="/",
        content=("Welcome to MagicWand Co.!\n"
                 "Dive into a world where magic meets reality. We specialize in crafting the finest magical artifacts.\n"
                 "- About Us [Clickable '/about']\n"
                 "- Our Products [Clickable '/products']\n"
                 "- Blog [Clickable '/blog']\n"
                 "- Contact Us [Clickable '/contact']\n"
                 "- Meet Our Team [Clickable '/team']\n"
                 "Discover the magic today!")
    ),
    # About page
    dict(
        name="/about",
        content=("About MagicWand Co.\n"
                 "Our journey began over a century ago, in a small enchanted forest. Today, we're leaders in the magical realm.\n"
                 "Meet Our Team [Clickable '/team']\n"
                 "Explore Our Products [Clickable '/products']\n"
                 "- CEO: Mr. Gandalf [Clickable '/about/team/ceo']\n"
                 "- Head of Engineering: Ms. Hermione [Clickable '/about/team/head_engineering']\n"
                 "- Chief Potion Master: Dr. Snape [Clickable '/about/team/chief_potion_master']\n"
                 "Our mission is to bring magic into every household.")
    ),
    # Contact page
    dict(
        name="/contact",
        content=("Contact Us at MagicWand Co.\n"
                 "We're here to answer all your magical inquiries. Whether you need help with a product or want to learn more about our offerings, reach out to us!\n"
                 "Email: contact@magicwand.co\n"
                 "Phone: +1-800-MAGICWAND\n"
                 "Follow us on WitcherGram @MagicWandCo")
    ),
    # Additional blog and subpages
    dict(
        name="/blog",
        content=("Our Magical Blog\n"
                 "Stay updated with the latest trends in the wizarding world. Our experts share insights, tips, and magical stories.\n"
                 "- Latest Potions [Clickable '/blog/potions']\n"
                 "- Wand Maintenance Tips [Clickable '/blog/wand-maintenance']\n"
                 "Join our community of magic enthusiasts!")
    ),
    dict(
        name="/blog/potions",
        content=("Potions Blog\n"
                 "From healing elixirs to love potions, our potion masters cover it all. Discover the secrets behind powerful concoctions and brews.\n"
                 "Featured Post: The Art of Brewing Invisibility Potions\n"
                 "Upcoming Event: Potion-Making Workshop - Enroll Now!")
    ),
    dict(
        name="/blog/wand_maintenance",
        content=("Wand Maintenance Blog\n"
                 "Your wand is your most loyal companion. Learn how to care for it and enhance its magical abilities.\n"
                 "Recent Article: 5 Signs Your Wand Needs Care\n"
                 "Expert Tips: Aligning Your Wand with Your Magical Energy")
    ),
    # CEO page
    dict(
        name="/about/team/ceo",
        content=("Mr. Gandalf - CEO\n"
                 "With decades of experience in the magical realm, Mr. Gandalf is a visionary leader. He's guided MagicWand Co. to new heights.\n"
                 "Email: gandalf.ceo@magicwand.co\n"
                 "Quote: 'Magic is not just in wands, it's within us.'")
    ),
    # Head of Engineering page
    dict(
        name="/about/team/head_engineering",
        content=("Ms. Hermione - Head of Engineering\n"
                 "Ms. Hermione is at the forefront of magical technology. Her innovations have revolutionized how we interact with magic.\n"
                 "Email: hermione.engineering@magicwand.co\n"
                 "Achievements: Creator of the Self-Replenishing Spellbook\n"
                 "Interests: Combining ancient spells with modern tech")
    ),
    # Chief Potion Master page
    dict(
        name="/about/team/chief_potion_master",
        content=("Dr. Snape - Chief Potion Master\n"
                 "A master of potions, Dr. Snape's expertise lies in creating the most effective and intricate brews.\n"
                 "Email: snape.potions@magicwand.co\n"
                 "Famous Creation: The Everlasting Elixir\n"
                 "Philosophy: 'In every potion, there is a story.'")
    )
]

# Initialize Faker with a fixed random seed for consistent results
Faker.seed(42)
fake = Faker()
random.seed(42)

def create_synthetic_pages():
    departments = ["Engineering", "Marketing", "Sales", "Customer Support"]
    products = ["Wand", "Potion", "Spellbook", "Crystal Ball"]
    adjectives = ["Mystical", "Enchanted", "Ancient", "Magical", "Legendary", "Rare"]

    metadata = {
        "item_info": [],
        "person_info": []
    }
    team_pages = []
    product_pages = []
    dept_to_team = {dept: [] for dept in departments}  # Dictionary to hold team members by department
    product_to_variants = {product: [] for product in products}  # Dictionary to hold product variants

    used_names = set()
    for department in departments:
        num_members = random.randint(3, 6)
        for _ in range(num_members):
            while True:
                member_name = fake.name()
                if member_name not in used_names:
                    used_names.add(member_name)
                    break
            
            page_name = f"/team/{department.lower().replace(' ', '_')}/{member_name.lower().replace(' ', '_')}"
            email = f"{member_name.lower().replace(' ', '.')}@{fake.domain_name()}"
            phone_number = fake.phone_number()
            expertise = '/'.join(random.sample(products, 2))
            content = (f"{member_name} of {department} Department\n"
                    f"Expertise: {expertise}\n"
                    f"Email: {email}\n"
                    f"Phone: {phone_number}")
            team_pages.append(dict(name=page_name, content=content))
            dept_to_team[department].append(page_name)
            
            metadata["person_info"].append(dict(
                name=member_name,
                expertise=expertise,
                email=email,
                phone_number=phone_number,
                department=department
            ))

    used_product_names = set()
    for product in products:
        num_variants = random.randint(3, 6)
        for _ in range(num_variants):
            while True:
                adjective = random.choice(adjectives)
                variant_name = f"{adjective} {product}"
                if variant_name not in used_product_names:
                    used_product_names.add(variant_name)
                    break
            page_name = f"/products/{product.lower().replace(' ', '_')}/{variant_name.lower().replace(' ', '_')}"
            original_price = None
            price = round(random.uniform(49.99, 199.99), 2)
            price_str = f"${price:.2f}"
            # Optional discount
            if random.random() < 0.1:
                off_pct = random.randint(10, 50)
                original_price = price
                price = round(price * (1 - off_pct/100), 2)
                price_str = f"${price} ({off_pct}% off!) -> Now at Price: ${price:.2f}"
            
            content = (
                f"Item: {variant_name}\n"
                f"Description: High-quality magical item.\n"
                f"Price: {price_str}"
            )
            product_pages.append(dict(name=page_name, content=content))
            product_to_variants[product].append(page_name)

            metadata["item_info"].append(dict(
                name=variant_name,
                price=price,
                original_price=original_price,
            ))

    # Aggregate Team Page
    team_aggregate_content = "Our Teams:\n"
    for dept, members in dept_to_team.items():
        team_aggregate_content += f"- {dept} Team [Clickable '/team/{dept.lower().replace(' ', '_')}']\n"
        
        member_agg_content = f"{dept} Team:\n"
        for member in members:
            member_agg_content += f"- {member} [Clickable '{member}']\n"
        team_pages.append(dict(
            name=f"/team/{dept.lower().replace(' ', '_')}",
            content=member_agg_content
        ))

    team_pages.append(dict(name="/team", content=team_aggregate_content))

    # Aggregate Product Page
    product_aggregate_content = "Our Products:\n"
    for product, variants in product_to_variants.items():
        product_aggregate_content += f"- {product} [Clickable '/products/{product.lower().replace(' ', '_')}']\n"

        variant_agg_content = f"{product}:\n"
        for variant in variants:
            variant_agg_content += f"- {variant} [Clickable '{variant}']\n"
        product_pages.append(dict(
            name=f"/products/{product.lower().replace(' ', '_')}",
            content=variant_agg_content
        ))

    product_pages.append(dict(name="/products", content=product_aggregate_content))

    return team_pages + product_pages, metadata

pages, metadata = create_synthetic_pages()
PAGES += pages

# Save into JSONL
with open(os.path.join(os.path.dirname(__file__), "web_pages.jsonl"), "w") as f:
    for page in PAGES:
        f.write(f"{json.dumps(page)}\n")
print(f"Saved {len(PAGES)} pages to web_pages.jsonl")

with open(os.path.join(os.path.dirname(__file__), "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to metadata.json")
