"""Generates kb/cards/*.json (all except the hand-written G_PLANTHOPPER.json) from the NIPHM/DPPQS AESA IPM
package text extracts in $AGROARMOUR_SOURCES. Written 2026-11-19.
Edit this script and re-run it rather than hand-editing card JSON, then run `make kb-lint`."""
import json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "kb" / "cards"
PUB = "NIPHM and Directorate of Plant Protection, Quarantine and Storage, Government of India"


def src(crop, sid):
    return {"id": sid, "title": f"Integrated Pest Management Package for {crop} (AESA based)", "publisher": PUB,
            "url": f"https://www.niphm.gov.in/IPMPackages/{crop}.pdf", "accessed": "2026-11-19", "local_text": f"{crop}.txt"}


def ch(active, form, dose, unit, sid="S1"):
    return {"active": active, "formulation": form, "dose": dose, "dose_unit": unit,
            "water_l_per_ha": None, "phi_days": None, "source_id": sid}


SAFE_PEST = ["Spray only when the pest count crosses the threshold. Wear gloves and a mask when you spray."]
SAFE_DIS = ["Spray only when the disease is seen on the crop. Wear gloves and a mask when you spray."]
SAFE_NONE = ["Do not spray chemicals without advice from your local agriculture officer."]


def card(kind, key, crops, en, sci, rel, identify, damage, mode, srcs, cult=(), bio=(), chem=(), etl=None, etl_sid="S1",
         safety=None, region=None, season=None, stage=None, also=None):
    cult, bio = list(cult), list(bio)
    th = {"etl": etl} if etl else {}
    if safety is None:
        safety = (SAFE_DIS if kind == "disease" else SAFE_PEST) if chem else SAFE_NONE
    names = {"en": en}
    if sci:
        names["scientific"] = sci
    c = {"id": f"{kind}.{key}", "kind": kind, "key": key}
    if also:
        c["also_keys"] = also
    c.update({
        "crops": crops, "names": names, "relevance_india": rel, "identify": identify, "damage": damage,
        "severity": {"mode": mode, "thresholds": th, "source_id": etl_sid if etl else None},
        "actions": {
            "low": {"cultural": cult[:2], "biological": bio, "chemical": []},
            "medium": {"cultural": cult, "biological": bio, "chemical": []},
            "high": {"cultural": cult, "biological": bio, "chemical": list(chem)},
        },
        "safety": safety, "region_notes": region or {}, "season_notes": season or {}, "stage_notes": stage or {},
        "sources": [src(cr, sid) for sid, cr in srcs],
    })
    (OUT / f"{key}.json").write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def short(kind, key, crops, en, srcs, generic, rel="low", sci=None, mode="presence_incidence", also=None, what="pest"):
    if kind == "disease":
        ident = [f"The photo model matched this to {en}.", "Ask an expert to confirm this disease."]
    else:
        ident = [f"The photo model matched this to {en}.", "This pest is not common in India. Ask an expert to confirm it."]
    card(kind, key, crops, en, sci, rel, ident, "", mode, srcs, cult=[generic], also=also)


RICE = [("S1", "Rice")]
GEN_POTATO = "Collect and destroy the insect infested plant parts."
GEN_WHEAT = "Collect and destroy the diseased and insect infected plant parts."
GEN_RICE = "Remove and destroy the diseased or pest infested plant parts."

# ---------------- RICE PESTS ----------------
card("pest", "G_RICE_LEAFFOLDER", ["rice"], "Rice leaf folder", "Cnaphalocrocis medinalis", "high",
     ["The rice leaf is folded and a small caterpillar lives inside the fold.",
      "The leaf folder is a major rice pest in India."],
     "", "presence_incidence", RICE,
     cult=["Plant early to avoid heavy leaf damage.", "Use wider spacing and a low dose of nitrogen fertilizer.",
           "Do not over-fertilize the field. Highly fertilized plots attract the moths.",
           "Remove the causes of shade in the field. Damage is higher in shaded areas."],
     bio=["Release Trichogramma japonicum or T. chilonis at 1 lakh/ha when you see egg masses or moths.",
          "Protect spiders and other natural enemies in the field."],
     chem=[ch("cartap hydrochloride", "50% SP", "1000", "g/ha"),
           ch("cartap hydrochloride", "4% granules", "18750-25000", "g/ha")],
     etl="Early to late tillering: 2 Fully damaged leaves (FDL) with larva/hill.",
     stage={"vegetative": "Act when you count 2 fully damaged leaves with larva per hill."})

card("pest", "G_RICE_CATERPILLAR", ["rice"], "Rice leaf-eating caterpillar (swarming caterpillar)", "Spodoptera mauritia", "medium",
     ["The photo model matched this to a caterpillar that eats rice leaves.",
      "The swarming caterpillar is a regional rice pest."],
     "", "presence_incidence", RICE,
     cult=["Do summer ploughing. It exposes the larvae and pupae to birds and weather.",
           "Collect the egg masses and larvae and destroy them.",
           "Drag a coir rope over the crop to dislodge the caterpillars."],
     bio=["Protect spiders, mirid bugs and other natural enemies in the field."],
     etl="Early to late tillering: 1 damaged tiller/hill or 2 larvae/ m2.",
     region={"OD": "The swarming caterpillar is a regional rice pest here.", "WB": "The swarming caterpillar is a regional rice pest here.",
             "JH": "The swarming caterpillar is a regional rice pest here.", "CG": "The swarming caterpillar is a regional rice pest here.",
             "PB": "The swarming caterpillar is a regional rice pest here."})

card("pest", "G_RICE_STEMBORER", ["rice"], "Rice stem borer", "Scirpophaga incertulas", "high",
     ["The photo model matched this to the rice stem borer.",
      "Dead heart in the young plant and white ear head are signs of stem borer damage.",
      "The yellow stem borer is a major rice pest in India."],
     "The larva bores into the stem.", "presence_incidence", RICE,
     cult=["Clip the tips of the seedlings before you transplant them.",
           "Plough the field after harvest and flood it to kill the larvae in the stubble.",
           "Harvest close to the ground to destroy the pest in the stubble.",
           "Install pheromone traps at 20 traps/ha to trap the male moths."],
     bio=["Release Trichogramma japonicum at 1 lakh/ha when you see egg masses or moths.",
          "Collect egg masses and keep them in a bamboo cage so that the egg parasites can escape."],
     chem=[ch("cartap hydrochloride", "4% granules", "18750", "g/ha"),
           ch("cartap hydrochloride", "50% SP", "1000", "g/ha")],
     etl="Early to late tillering: 2 egg-mass/m2 or 10% dead heart or 1 moth/m2 or 25 moths/ trap/week.",
     stage={"seedling": "In the nursery, act when you see 1 egg-mass/m2."}, also=["rice__dead_heart"])

card("pest", "G_RICE_GALL_MIDGE", ["rice"], "Rice gall midge", "Orseolia oryzae", "high",
     ["The photo model matched this to the rice gall midge.",
      "The silver shoot is a sign of gall midge damage.",
      "The gall midge is a major rice pest in India."],
     "", "presence_incidence", RICE,
     cult=["Drain the rice field for 5-7 days.", "Control grassy weeds and wild rice around the field.",
           "Plant early and use early maturing varieties.",
           "Use only moderate amounts of nitrogen and potassium fertilizers in split doses.",
           "Plough under the ratoon of the previous crop.",
           "Complete the planting in an area within 3 weeks."],
     chem=[ch("carbosulfan", "25% EC", "800-1000", "ml/ha"),
           ch("carbosulfan", "6% G", "16700", "g/ha")],
     etl="Early to late tillering: 1 gall/m2 or 10% Silver shoot.")

short("pest", "G_RICE_WATER_WEEVIL", ["rice"], "rice water weevil", RICE, GEN_RICE, sci="Lissorhoptrus oryzophilus")

card("pest", "G_LEAFHOPPER", ["rice", "cotton"], "Green leafhopper", "Nephotettix spp.", "high",
     ["The photo model matched this to a leafhopper.",
      "Green leafhoppers spread virus diseases of rice."],
     "", "count", [("S1", "Rice"), ("S2", "Cotton")],
     cult=["Plant early and do not plant at the time of peak leafhopper activity.",
           "Transplant older seedlings. They are less open to virus disease.",
           "Keep the field and the bunds free of weeds.",
           "Apply nitrogen at the optimal level."],
     bio=["Protect natural enemies in the field."],
     etl="Cotton: More than 2 leaf hopper per leaf and appearance of crinkling and curling of few leaves in the lower portion of plant + marginal yellowing of leaves.",
     etl_sid="S2")

# ---------------- RICE hispa / diseases ----------------
card("pest_damage", "rice__hispa", ["rice"], "Rice hispa", "Dicladispa armigera", "high",
     ["The photo model matched this to damage by the rice hispa beetle.",
      "The rice hispa is a regional rice pest."],
     "", "presence_incidence", RICE,
     cult=["Clip the top three-fourths of the leaves of a badly infested young crop. This removes the eggs and grubs.",
           "Collect the adults with a sweep net and destroy them.",
           "Remove rice ratoons and volunteer rice in the crop-free season.",
           "Do not top-dress nitrogen fertilizer when hispa attack is high."],
     chem=[ch("chlorpyrifos", "20% EC", "1250", "ml/ha"),
           ch("quinalphos", "25% gel", "1000", "ml/ha")],
     etl="Early to late tillering: 2 adults or 2 dead leaf /hill.",
     region={r: "The rice hispa is a pest of rice in this state." for r in ["BR", "WB", "AS", "OD", "PB", "HP", "UP", "UK"]})

card("disease", "rice__blast", ["rice"], "Rice blast", "Pyricularia oryzae", "high",
     ["The photo model matched this to rice blast.", "Blast is a major rice disease in India."],
     "", "lesion_area", RICE,
     cult=["Use a resistant or moderately resistant variety.", "Use disease free pure seed.",
           "Use fertilizers in a balanced way as per local advice.",
           "Remove and destroy the diseased plant parts."],
     chem=[ch("tricyclazole", "70% WG", "300", "g/ha"),
           ch("isoprothiolan", "40% EC", "750", "ml/ha"),
           ch("carbendazim", "50% WP", "250-500", "g/ha")],
     etl="Early to late tillering: Foliar blast 3-5 lesions/leaf. Panicle initiation to booting: Neck blast 2-5 neck infected plants/m2.")

card("disease", "rice__brown_spot", ["rice"], "Rice brown spot", "Helminthosporium oryzae", "high",
     ["The photo model matched this to rice brown spot.", "Brown spot is a major rice disease in India."],
     "", "lesion_area", RICE,
     cult=["Use a resistant or moderately resistant variety.", "Use disease free pure seed.",
           "Use fertilizers in a balanced way as per local advice.",
           "Remove and destroy the diseased plant parts."],
     etl="Early to late tillering: Brown spot 2-3 spots/leaf & 2-3 infected plants/ m2.")

card("disease", "rice__bacterial_leaf_blight", ["rice"], "Bacterial leaf blight of rice", "Xanthomonas oryzae pv. oryzae", "high",
     ["The photo model matched this to bacterial leaf blight of rice.",
      "Bacterial leaf blight is a major rice disease in India."],
     "", "lesion_area", RICE,
     cult=["Reduce the nitrogen fertilizer. If needed, apply only a small dose of nitrogen in more split doses.",
           "Use alternate wetting and drying. Do not let the water stand in the field.",
           "Use a resistant variety.",
           "Use disease free pure seed."],
     chem=[ch("streptomycin sulphate + tetracycline hydrochloride", "9% + 1% SP", "100-150", "ppm")],
     etl="Early to late tillering: BLB 2-3 infected leaves/m2.")

card("disease", "rice__tungro", ["rice"], "Rice tungro disease", "Rice tungro virus", "medium",
     ["The photo model matched this to rice tungro disease.",
      "Tungro is a virus disease. Green leafhoppers spread it."],
     "", "lesion_area", RICE,
     cult=["Use a resistant variety.", "Transplant older seedlings. They are less open to virus disease.",
           "Plant early. Do not plant at the time of peak leafhopper activity.",
           "Keep the field and the bunds free of weeds."],
     etl="Early to late tillering: 1 Tungro infected plants/m2 & 2 GLH/hill (in fungus endemic areas).")

short("disease", "rice__downy_mildew", ["rice"], "downy mildew of rice", RICE, GEN_RICE, rel="low", mode="lesion_area")
short("disease", "rice__bacterial_leaf_streak", ["rice"], "bacterial leaf streak of rice", RICE, GEN_RICE, rel="medium", mode="lesion_area")
short("disease", "rice__bacterial_panicle_blight", ["rice"], "bacterial panicle blight of rice", RICE, GEN_RICE, rel="low", mode="presence_incidence")

# ---------------- MAIZE ----------------
MZ = [("S1", "Maize")]
card("pest", "G_MAIZE_BORER", ["maize"], "Maize stem borer", "Chilo partellus", "high",
     ["The photo model matched this to the maize stem borer.",
      "Dead heart in the young plant is a sign of stem borer damage.",
      "The maize stem borer is a major maize pest in India."],
     "", "presence_incidence", MZ,
     cult=["Do deep summer ploughing and then leave the field fallow.",
           "Grow a legume as an intercrop with maize. Maize-Soybean, Maize-Cowpea and Maize-Green gram are good examples.",
           "Remove the dead hearts. This reduces the second generation attack.",
           "Use fertilizers in a balanced way."],
     bio=["Release Trichogramma chilonis at 1,60,000 /ha on 7 and 15 days old crop and again if required.",
          "Protect Cotesia flavipes, spiders and other natural enemies in the field."])

card("disease", "maize__common_rust", ["maize"], "Common rust of maize", "Puccinia sorghi", "high",
     ["The photo model matched this to common rust of maize.",
      "Common rust is a major maize disease in India."],
     "", "lesion_area", MZ,
     cult=["Use certified seed of a recommended variety.", "Remove the crop debris from the field.",
           "Use crop rotation.", "Use a balanced dose of fertilizer. Avoid a high level of nitrogen."],
     chem=[ch("zineb", "75% WP", "1.5-2", "kg/ha")])

card("disease", "maize__northern_leaf_blight", ["maize"], "Northern leaf blight of maize (Turcicum leaf blight)",
     "Exserohilum turcicum", "high",
     ["The photo model matched this to northern leaf blight of maize.",
      "This disease is also called Turcicum leaf blight. It is a major maize disease in India."],
     "", "lesion_area", MZ,
     cult=["Choose a field with good drainage.", "Remove the crop debris from the field.",
           "Use crop rotation.", "Use a balanced dose of fertilizer. Avoid a high level of nitrogen.",
           "Use certified seed of a recommended variety."],
     etl="Leaf blight 10% incidence.")

short("disease", "maize__gray_leaf_spot", ["maize"], "gray leaf spot of maize", MZ,
      "Rogue and destroy the infected plants.", rel="medium", mode="lesion_area")

# ---------------- TOMATO ----------------
TM = [("S1", "Tomato")]
card("disease", "tomato__early_blight", ["tomato"], "Early blight of tomato", "Alternaria solani", "high",
     ["Small black spots appear on the older leaves.",
      "The spots grow and show rings like a bull's eye. The tissue around the spot may turn yellow."],
     "The disease can kill much of the leaves in hot and humid weather. It also spots the stem and the fruit.",
     "lesion_area", TM,
     cult=["Use resistant or tolerant varieties.", "Change the nursery bed location every season.",
           "Remove weeds and volunteer tomato plants.", "Do not plant overlapping crops in the adjacent area."],
     chem=[ch("azoxystrobin", "23% SC", "200", "ml in 200 l of water/acre"),
           ch("pyraclostrobin", "20% WG", "150-200", "g in 200 l of water/acre"),
           ch("mancozeb", "75% WG", "400", "g in 200 l of water/acre")],
     region={r: "Early blight is reported on tomato in this state." for r in ["OD", "HR", "HP", "RJ", "PB", "UP", "GJ", "MH", "AP", "KA", "TN", "MP", "BR"]})

card("disease", "tomato__late_blight", ["tomato"], "Late blight of tomato", "Phytophthora infestans", "high",
     ["The photo model matched this to late blight of tomato.",
      "Late blight is a major tomato disease."],
     "", "lesion_area", TM,
     cult=["Do field sanitation.", "Destroy the alternate host plants.", "Give light irrigation."],
     chem=[ch("azoxystrobin", "23% SC", "200", "ml in 200 l of water/acre"),
           ch("cymoxanil + mancozeb", "8% + 64% WP", "600", "g in 200-300 l of water/acre"),
           ch("famoxadone + cymoxanil", "16.6% + 22.1% SC", "200", "g in 200 l of water/acre")],
     region={r: "Late blight is reported on tomato in this state." for r in ["OD", "HP", "RJ", "UP", "KA", "TN"]})

card("disease", "tomato__bacterial_spot", ["tomato"], "Bacterial spot of tomato", "Xanthomonas campestris pv. vesicatoria", "high",
     ["Small brown water soaked round spots with a yellow halo appear on the leaves.",
      "Small water soaked spots appear on green fruit. Later they become raised and scabby."],
     "Older leaves can lose many leaflets in moist weather.", "lesion_area", TM,
     cult=["Remove and destroy the plant debris.", "Remove the weeds and volunteer tomato plants.",
           "Use seedlings from healthy seed beds."],
     chem=[ch("streptomycin sulfate + tetracycline hydrochloride", "9% + 1% SP", "40-100", "ppm")],
     stage={"seedling": "Give two sprays. Give one before transplanting in the seed bed and one after transplanting in the main field."},
     region={r: "Bacterial leaf spot is reported on tomato in this state." for r in ["AP", "TN", "KA"]})

card("disease", "tomato__septoria_leaf_spot", ["tomato"], "Septoria leaf spot of tomato", "Septoria lycopersici", "medium",
     ["Small round to irregular spots with a grey center and a dark edge appear on the leaves.",
      "The spots start on the lower leaves and move upwards."],
     "The spots join and the leaves dry up and fall. Fruits are rarely attacked.", "lesion_area", TM,
     cult=["Remove and destroy the affected plant parts."],
     chem=[ch("mancozeb", "75% WP", "600-800", "g in 300 l of water/acre")])

card("disease", "tomato__yellow_leaf_curl_virus", ["tomato"], "Tomato leaf curl virus", "Tomato leaf curl virus", "high",
     ["The plant is small and the leaves roll downward and crinkle.",
      "New leaves show some yellow colour and later they curl.",
      "The whitefly carries this virus from plant to plant."],
     "Older leaves become leathery and brittle. The plant is pale and bushy.", "presence_incidence", TM,
     cult=["Raise the nursery under a net with a fine mesh so that the whitefly cannot enter.",
           "Select seed from healthy plants.", "Remove the infected plants carefully and destroy them.",
           "Do not transplant seedlings that have the virus disease.",
           "Raise barrier crops such as maize or sorghum around the field."],
     chem=[ch("imidacloprid", "17.8% SL", "60-70", "ml in 200 l of water/acre")],
     also=["tomato__leaf_curl_virus"],
     region={r: "Leaf curl virus is reported on tomato in this state." for r in ["MP", "BR", "AS", "OD", "WB", "HR", "HP", "RJ", "PB", "UP", "GJ", "MH", "AP", "KA", "TN"]})

card("disease", "tomato__mosaic_virus", ["tomato"], "Tomato mosaic virus", "Tomato mosaic virus", "medium",
     ["The leaves show light and dark green patches.",
      "The leaflets are twisted, puckered and smaller than normal.",
      "The plant is small, pale green and thin."],
     "The virus spreads by touch, by clothes, by hands, by plant debris and by tools.", "presence_incidence", TM,
     cult=["Select seed from healthy plants.", "Rinse the seeds well and dry them in shade.",
           "Remove the infected plants carefully and destroy them.",
           "Do not transplant seedlings that have the virus disease.",
           "Do crop rotation with crops other than tobacco, potato, chilli, capsicum and brinjal."])

short("disease", "tomato__leaf_mold", ["tomato"], "leaf mold of tomato", TM,
      "Remove and destroy the affected plant parts.", rel="medium", mode="lesion_area")

# ---------------- POTATO ----------------
PT = [("S1", "Potato")]
card("disease", "potato__early_blight", ["potato"], "Early blight of potato", "Alternaria solani", "high",
     ["Small black spots appear on the older leaves.",
      "The spots grow and show rings like a bull's eye. The tissue around the spot may turn yellow."],
     "Much of the leaves can be killed in hot and humid weather. Spots on the stem can girdle the plant.",
     "lesion_area", PT,
     cult=["Collect and destroy the crop debris.", "Use fertilizers in a balanced way.",
           "Avoid water logging.", "Collect and destroy the diseased plant parts."],
     chem=[ch("mancozeb", "75% WP", "600-800", "g in 300 l of water/acre"),
           ch("captan", "75% WP", "666", "g in 400 l of water/acre"),
           ch("hexaconazole", "2% SC", "1.2", "l in 200 l of water/acre")],
     stage={"vegetative": "The disease is seen mostly during tuber bulking and goes on until harvest."})

card("disease", "potato__late_blight", ["potato"], "Late blight of potato", "Phytophthora infestans", "high",
     ["Leaves look blistered as if scalded by hot water. Later they turn brown or black and dry out.",
      "Spots on the underside of the leaves look like flour.",
      "The stems blacken from the tips and dry out."],
     "The disease acts very quickly. Tubers get dry brown spots on the skin and flesh. Infected plants can die within two or three days.",
     "lesion_area", PT,
     cult=["Use short-duration varieties.", "Collect and destroy the crop debris.", "Avoid water logging.",
           "Collect and destroy the diseased plant parts."],
     chem=[ch("azoxystrobin", "23% SC", "200", "ml in 200 l of water/acre"),
           ch("dimethomorph", "50% WP", "400", "g in 300 l water/acre"),
           ch("mancozeb", "75% WP", "600-800", "g in 300 l water/acre")])

# ---------------- COMMON INSECT PESTS ----------------
card("pest", "G_WHITE_GRUB", ["groundnut", "potato"], "White grub", "Holotrichia spp.", "high",
     ["The grub is large and shaped like the letter C.",
      "The head is hard and brown. The grub has three pairs of legs."],
     "The grubs feed on the roots and then on the tubers. They make irregular holes in the potato tubers.",
     "presence_incidence", [("S1", "Groundnut"), ("S2", "Potato")],
     cult=["Do deep ploughing in summer to expose the grubs.",
           "Collect and destroy the white grub beetles from jujube or neem trees around the field after the early rains.",
           "Sow in time with the pre-monsoon showers."],
     bio=["Use entomopathogenic nematodes (EPN) at one billion per acre in infested fields."],
     chem=[ch("chlorpyrifos", "20% EC (seed treatment)", "2.5 to 12.5", "ml/kg seed")],
     etl="Groundnut: 1 grub/m2.")

card("pest", "G_CUTWORM", ["potato", "tomato"], "Cutworm", "Agrotis ipsilon", "high",
     ["The full-grown larva is dark or dark brown. It has a plump and greasy body.",
      "The larva has striped markings along the sides of its body and lives in the soil."],
     "", "presence_incidence", [("S1", "Potato"), ("S2", "Tomato")],
     cult=["Place heaps of rice bran in several places in the late afternoon. Remove the larvae from the bran next day and destroy them.",
           "Flood the field before planting when you can.",
           "Do deep summer ploughing.", "Use well decomposed organic manure.",
           "Collect the caterpillars from the soil around the plants in a small area."])

card("pest", "G_ARMYWORM", ["maize", "wheat", "rice"], "Armyworm", "Mythimna separata", "high",
     ["The caterpillar has stripes along the body.",
      "It feeds at dawn and dusk because it avoids sunlight."],
     "The main sign is loss of leaves. The larvae chew the leaves from the edges. They can cut the neck of the cereal head.",
     "presence_incidence", [("S1", "Maize"), ("S2", "Wheat"), ("S3", "Rice")],
     cult=["Check the field often for the pest and its natural enemies.",
           "Grow barrier crops like mustard around the field.",
           "Erect bird perches at 10/ acre for predatory birds.",
           "Do summer ploughing to expose the larvae and pupae to birds."],
     bio=["Protect natural enemies in the field."],
     etl="Maize: Armyworm One larva per plant.")

card("pest", "G_APHID", ["wheat", "potato", "groundnut"], "Aphids", "Sitobion avenae", "high",
     ["Aphids are small, soft-bodied, pear-shaped green insects.",
      "They appear in large numbers on young leaves or ears in cold and cloudy weather."],
     "Aphids suck the sap of the plant. They also spread virus diseases from plant to plant.",
     "count", [("S1", "Wheat"), ("S2", "Potato"), ("S3", "Groundnut")],
     cult=["Plant tall border crops like maize, sorghum or millet to reduce the pest.",
           "Grow barrier crops like mustard around the field.",
           "Use healthy seed.", "Check the field often for the pest and its natural enemies."],
     bio=["Spray neem seed kernel extract (NSKE) 5%.",
          "Protect ladybird beetles, lacewings, spiders and hover flies in the field."],
     chem=[ch("thiamethoxam", "25% WG", "20", "g in 200 l of water per acre", "S1"),
           ch("imidacloprid", "17.8% SL", "100-125", "ml/500 lit. of water", "S3")])

card("pest", "G_SPIDER_MITE", ["tomato", "potato"], "Spider mite", "Tetranychus spp.", "medium",
     ["Very small red mites live on the leaves.", "The leaves turn reddish brown and bronze in colour."],
     "Heavy attack makes silken webbing on the leaves. The leaves wither and dry. Flower and fruit formation is affected.",
     "count", [("S1", "Tomato"), ("S2", "Potato")],
     cult=["Do field sanitation and remove the affected plants.", "Plant tall border crops such as maize and sorghum."],
     bio=["Protect predatory mites, ladybird beetles and lacewings in the field."],
     chem=[ch("spiromesifen", "22.9% SC", "250", "ml in 200 l of water/acre")],
     also=["tomato__spider_mites"])

card("pest", "G_WHITEFLY", ["tomato", "potato", "cotton"], "Whitefly", "Bemisia tabaci", "high",
     ["The adult is a small insect with a light yellow body and pure white wings.",
      "The adults are found on the underside of the leaves."],
     "The leaves show spots, turn yellow, curl downward and dry. The whitefly spreads the tomato leaf curl disease.",
     "count", [("S1", "Tomato"), ("S2", "Cotton")],
     cult=["Use yellow sticky traps or cards at 10/acre.", "Raise barrier crops such as maize or sorghum around the field.",
           "Do field sanitation and remove the affected plants."],
     bio=["Spray neem seed kernel extract (NSKE) 5%.", "Protect the parasitoids and the predators of the whitefly in the field."],
     chem=[ch("imidacloprid", "17.8% SL", "60-70", "ml in 200 l of water/acre"),
           ch("thiamethoxam", "25% WG", "80", "g in 200 l of water/acre"),
           ch("spiromesifen", "22.9% SC", "250", "ml in 200 l of water/acre")],
     etl="Cotton: More than 10 whiteflies found in middle region of the plant in >50% (two out of four) of plants.",
     etl_sid="S2")

card("pest", "G_TOBACCO_CATERPILLAR", ["tomato", "groundnut"], "Tobacco caterpillar", "Spodoptera litura", "high",
     ["The young caterpillars live together and scrape the leaf. The leaf looks papery white.",
      "Later the caterpillars make irregular holes in the leaves."],
     "Heavy attack removes most of the leaves. The caterpillars also make holes in the fruits.",
     "presence_incidence", [("S1", "Tomato"), ("S2", "Groundnut")],
     cult=["Do field sanitation.", "Install pheromone traps at 4-5/acre to monitor the moths.",
           "Grow castor as a trap crop for egg laying.", "Erect bird perches at 20/acre for predatory birds."],
     bio=["Release the egg parasitoid Trichogramma pretiosum at 20,000/acre/week four times.",
          "Spray NSKE 5% against the eggs and first instar larvae."],
     chem=[ch("indoxacarb", "14.5% SC", "160 - 200", "ml in 120-240 l of water/acre"),
           ch("novaluron", "10 % EC", "300", "ml in 200-400 l of water/acre"),
           ch("lambda-cyhalothrin", "4.9% CS", "120", "ml in 200 l of water/acre")],
     etl="Groundnut: Spodoptera / Helicoverpa 2 larvae/plant or 20-25% defoliation at 40 days.", etl_sid="S2",
     region={"OD": "The tobacco caterpillar is reported on tomato in this state.", "KA": "The tobacco caterpillar is reported on tomato in this state."})

card("pest", "G_THRIPS", ["wheat", "groundnut"], "Thrips", "Anaphothrips favicinctus", "high",
     ["Thrips are very small brown or black insects with narrow fringed wings.",
      "Heavy attack makes the leaf tissue look silver."],
     "The thrips feed on the sheath of the flag leaf, the stem and the heads.",
     "count", [("S1", "Wheat"), ("S2", "Groundnut")],
     cult=["Use blue sticky traps for thrips at 4-5 traps/acre.", "Collect and destroy the diseased and insect infected plant parts."],
     bio=["Protect predatory mites, predatory thrips and hover flies in the field."],
     etl="Groundnut: 5 adults/terminal buds.", etl_sid="S2")

card("pest", "G_FRUIT_FLY", ["mango"], "Fruit fly", "Bactrocera dorsalis", "high",
     ["The female fly puts eggs inside the skin of ripe fruit.",
      "The maggots feed on the fruit pulp and the fruit rots."],
     "The infested fruit rots.", "presence_incidence", [("S1", "Mango")],
     cult=["Collect the infested and fallen fruits and dispose of them. Do this 30-40 days before harvest.",
           "Plough the orchard in November-December to expose the pupae to the heat of the sun.",
           "Collect and destroy the adult flies."],
     bio=["Use methyl eugenol pheromone traps at 4/acre."])

card("pest", "G_MIRID_BUG", ["cotton"], "Mirid bug", "Creontiades biseratense", "medium",
     ["Mirid nymphs look like aphids but they move much faster.",
      "The adults and nymphs have long and thin antennae."],
     "Feeding kills the plant tips of young plants. Fed squares dry and fall. Fed bolls show sunken black spots and the seeds are shrunken and stained.",
     "count", [("S1", "Cotton")],
     cult=["Check the plant tips and squares often."],
     bio=["Protect lady beetles and lacewings in the field."])

# ---------------- NOT FOUND IN ANY SOURCE: short cards ----------------
short("pest", "G_MOLE_CRICKET", ["potato"], "mole cricket", [("S1", "Potato")], GEN_POTATO)
short("pest", "G_WIREWORM", ["potato"], "wireworm", [("S1", "Potato")], GEN_POTATO)
short("pest", "G_FLEA_BEETLE", ["potato"], "flea beetle", [("S1", "Potato")], GEN_POTATO)
short("pest", "G_BEET_WEEVIL", ["potato"], "beet weevil", [("S1", "Potato")], GEN_POTATO)
short("pest", "G_BLISTER_BEETLE", ["wheat"], "blister beetle", [("S1", "Wheat")], GEN_WHEAT)
short("pest", "G_SAWFLY", ["wheat"], "sawfly", [("S1", "Wheat")], GEN_WHEAT)
short("pest", "G_LOCUST", ["wheat"], "locust", [("S1", "Wheat")], GEN_WHEAT)
short("pest", "G_BUDWORM", ["wheat"], "budworm", [("S1", "Wheat")], GEN_WHEAT)
short("pest", "G_SLUG_CATERPILLAR", ["mango"], "slug caterpillar", [("S1", "Mango")],
      "Collect and destroy the plant parts infested with insect pests.")
print("done")
