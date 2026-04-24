"""
Generate expanded limitation test set with 20+ queries per limitation.
Then verify all Cranfield query IDs and doc IDs against ground truth.
"""
import json, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAN = os.path.join(BASE, "cranfield")

with open(os.path.join(CRAN, "cran_queries.json"), encoding="utf-8") as f:
    queries_raw = json.load(f)
qmap = {q["query number"]: q["query"] for q in queries_raw}

with open(os.path.join(CRAN, "cran_qrels.json"), encoding="utf-8") as f:
    qrels_raw = json.load(f)
qrels = {}
for e in qrels_raw:
    qn = int(e["query_num"])
    qrels.setdefault(qn, []).append({"id": e["id"], "position": e["position"]})

def cq(qnum, why):
    """Build a cranfield query entry with auto-populated text and docs."""
    docs = sorted(set(e["id"] for e in qrels.get(qnum, [])))
    return {"query_number": qnum, "query": qmap.get(qnum, "MISSING"), "relevant_docs": docs, "why": why}

def custom(cid, query, concept, docs, why):
    return {"custom_id": cid, "query": query, "target_concept": concept, "expected_relevant_docs": docs, "why": why}

# ============================================================
# 1. SEMANTIC UNDERSTANDING
# ============================================================
sem_cran = [
    cq(5, "chemical kinetic system vs reacting gas mixtures/dissociation"),
    cq(94, "heat transfer rate at stagnation point: conceptual gap with BL theory"),
    cq(4, "chemically reacting gas mixtures: domain-specific phrasing mismatch"),
    cq(73, "similitude laws vs similarity parameters; out of equilibrium vs nonequilibrium"),
    cq(155, "ablation measurement vs mass loss/heat protection shield"),
    cq(156, "ablation materials research: qualitative/quantitative vs specific material terms"),
    cq(201, "nonequilibrium chemical constituents: specialized concept vocabulary"),
    cq(53, "ionized gas + magnetic field: semantic gap across physics domains"),
    cq(20, "joule heating + magnetohydrodynamic: concept-level gap"),
    cq(10, "real-gas transport properties: conceptual domain mismatch"),
    cq(19, "dynamics of re-entry: realistic effects vs specific aerodynamic terms"),
    cq(74, "dissociated free stream + hypersonic simulation: concept bridging needed"),
    cq(208, "drag polar of lifting spacecraft: paraphrase of aerodynamic concepts"),
]
sem_custom = [
    custom("sem_c1", "how does thermal erosion protect spacecraft during atmospheric re-entry",
           "ablation heat protection", ["82"], "thermal erosion = ablation synonym"),
    custom("sem_c2", "air resistance effects on orbiting body altitude decay",
           "satellite orbit contraction under drag", ["613","614","615","548"], "air resistance = drag"),
    custom("sem_c3", "computing fluid shear forces on aircraft surfaces at high velocity",
           "skin friction at supersonic speeds", ["50","49","9","16"], "fluid shear forces = skin friction"),
    custom("sem_c4", "methods to estimate temperature distribution in multi-component structural panels under rapid heating",
           "transient thermal analysis in composites", ["5","6","29","30","66"], "multi-component panels = composite slabs"),
    custom("sem_c5", "mathematical modeling of oscillatory structural instability in thin panels exposed to airflow",
           "panel flutter analysis", ["15","52"], "oscillatory structural instability = flutter"),
    custom("sem_c6", "predicting nose-region thermal loads on vehicles entering planetary atmospheres",
           "stagnation point heat transfer on re-entry bodies", ["24","283"], "thermal loads = heat transfer"),
    custom("sem_c7", "electrically conducting fluid flow control using magnetic fields",
           "magnetohydrodynamic flow over plates", ["61","33","34"], "conducting fluid + magnetic = MHD"),
    custom("sem_c8", "numerical approaches for solving viscous fluid motion equations",
           "Navier-Stokes solutions", ["1078","1083","1084","1063"], "viscous fluid motion equations = Navier-Stokes"),
    custom("sem_c9", "structural failure under repeated cyclic loading in aerospace components",
           "fatigue of aircraft structures", ["75","724","726","727"], "cyclic loading failure = fatigue"),
    custom("sem_c10", "gas molecule mean free path effects on surface heat exchange",
           "slip flow heat transfer / rarefied gas effects", ["21","22"], "mean free path effects = slip flow"),
]

# ============================================================
# 2. HIGH DIMENSIONALITY / COST
# ============================================================
dim_cran = [
    cq(92, "extremely long verbose query about sonic boom minimization"),
    cq(114, "long multi-sentence query about airforces on oscillating wing"),
    cq(137, "verbose: time-to-failure + creep collapse + cylindrical shell"),
    cq(124, "multi-part query mixing viscous channel flow + conical shells"),
    cq(107, "long: summarizing theoretical/experimental work in noise environment"),
    cq(144, "long: membrane cylinders + internal pressure + shape maintenance"),
    cq(99, "verbose: uncontrolled vehicle tumbling in atmosphere"),
    cq(160, "long: torispherical shell + compressive stresses + buckling + internal pressure"),
    cq(104, "verbose: load deformation + inelastic material + non-uniform temperature"),
    cq(7, "long: ogive forebody + zero angle of attack + lower surface pressures + equivalent"),
    cq(46, "verbose: combined effect of surface heat and mass transfer on hypersonic flow"),
    cq(55, "verbose: incompressible BL info applied to compressible BL problems"),
]
dim_custom = [
    custom("dim_c1",
           "theoretical and experimental investigations of the combined effects of aerodynamic heating structural deformation boundary layer transition and surface roughness on the performance of supersonic aircraft wing panels during prolonged high-speed flight",
           "aerothermoelastic effects on panels", ["12","15","31","66","29"],
           "extremely long query: noise terms dilute TF-IDF signal"),
    custom("dim_c2",
           "analytical and numerical methods for solving the complete set of Navier-Stokes equations including viscous terms turbulence modeling and boundary conditions for incompressible and compressible flows over various body shapes at different Reynolds and Mach numbers",
           "Navier-Stokes solution methods", ["1078","1083","1084"],
           "verbose query with many redundant terms"),
    custom("dim_c3",
           "what are all the available experimental data and theoretical analyses and computational studies dealing with the effects of initial geometric imperfections and residual stresses and material nonlinearities on the elastic and inelastic buckling behavior of thin-walled circular cylindrical shells subjected to axial compression",
           "shell buckling with imperfections", ["821","822","824","760"],
           "extremely verbose: 40+ word query"),
    custom("dim_c4",
           "comprehensive investigation of laminar and turbulent boundary layer flows including effects of pressure gradient heat transfer surface roughness compressibility and three-dimensionality on flat plates cones and bodies of revolution at subsonic transonic supersonic and hypersonic speeds",
           "boundary layer comprehensive", ["50","49","54","55","62"],
           "kitchen-sink query: too many dimensions"),
    custom("dim_c5",
           "what methods either exact or approximate have been developed and validated for predicting the aerodynamic forces and moments acting on various wing planforms oscillating in unsteady transonic and supersonic flow fields",
           "unsteady aero forces on wings", ["900","902","916","919"],
           "verbose query needing dimensionality reduction"),
    custom("dim_c6",
           "investigation of thermal stresses discontinuity stresses edge loading effects and pressure-induced deformations in pressurized cylindrical and spherical shell structures used in aerospace vehicles",
           "stresses in pressurized shells", ["1038","1039","1041","1042","954"],
           "verbose multi-concept query"),
    custom("dim_c7",
           "determining the flutter speed boundaries vibration mode shapes natural frequencies and aeroelastic stability characteristics of heated supersonic wing panels with various boundary conditions and material properties",
           "panel flutter at supersonic speeds", ["15","391","858","859","948"],
           "verbose query spanning multiple sub-topics"),
    custom("dim_c8",
           "all papers reports and experimental data concerning the measurement prediction and control of transition from laminar to turbulent flow on aircraft surfaces at speeds ranging from low subsonic through transonic to hypersonic",
           "boundary layer transition survey", ["7","8","9","40","41","79","80","608"],
           "extremely broad verbose survey query"),
]

# ============================================================
# 3. SCALABILITY
# ============================================================
scale_cran = [
    cq(23, "broad: unsteady aerodynamics, 33 relevant docs"),
    cq(84, "broad topic: heat transfer to conical bodies"),
    cq(157, "very broad: flow fields for blunt-nosed bodies, 40+ rel docs"),
    cq(65, "broad: boundary layer on flat plate in shear flow, many docs"),
    cq(47, "broad: hypersonic viscous interactions over insulated flat plate"),
    cq(67, "broad: series expansions for BL on flat plate in shear flow"),
    cq(132, "broad: theoretical studies of creep buckling"),
    cq(225, "broad: design factors for lift-drag ratios above Mach 5"),
    cq(220, "broad: calculation procedure for all incompressible laminar BL flows"),
    cq(221, "broad: papers on calculation procedures for laminar incompressible flow"),
]
scale_custom = [
    custom("scale_c1", "all available experimental and theoretical papers on boundary layer behavior",
           "broad boundary-layer literature", ["2","3","4","7","8","9","16","40","41","50","54","55","62","63","73","80"],
           "extremely broad: hundreds of possible matches"),
    custom("scale_c2", "comprehensive survey of heat transfer research in aerospace applications",
           "heat transfer survey", ["5","6","21","22","24","29","37","49","62","66","81"],
           "broad survey query spanning many clusters"),
    custom("scale_c3", "survey of structural stability and buckling analysis methods for shells",
           "shell buckling survey", ["821","822","824","820","823","825","760"],
           "broad structural topic, many related docs"),
    custom("scale_c4", "overview of all supersonic and hypersonic flow research",
           "supersonic/hypersonic flow survey", ["25","27","35","36","44","56","57","20"],
           "extremely broad topic"),
    custom("scale_c5", "papers on aerodynamic heating and thermal protection",
           "aerodynamic heating survey", ["5","6","12","29","30","31","66","82"],
           "broad: many documents about heating"),
    custom("scale_c6", "all research on vibration and flutter of aerospace structures",
           "structural dynamics survey", ["14","15","42","52","390","391","858","859"],
           "broad structural dynamics topic"),
    custom("scale_c7", "experiments on compressible flow including shock waves and expansion waves",
           "compressible flow experiments", ["64","65","71","72","74"],
           "broad experimental topic"),
    custom("scale_c8", "theoretical work on inviscid flow past bodies at supersonic speeds",
           "inviscid supersonic flow theory", ["20","25","27","28","35","36","56","57"],
           "broad theory survey"),
    custom("scale_c9", "research on cylindrical shell behavior under various loading conditions",
           "cylindrical shell research", ["838","839","840","841","842","843"],
           "broad structural topic"),
    custom("scale_c10", "compilation of magnetohydrodynamic flow studies",
           "MHD studies survey", ["33","34","61","87","88","267","268"],
           "broad MHD topic"),
]

# ============================================================
# 4. WORD SENSE AMBIGUITY
# ============================================================
ambig_cran = [
    cq(85, "transition ambiguous: BL transition vs general"),
    cq(100, "elastic ambiguous; buckling spans structural/flow"),
    cq(39, "transition needs context to disambiguate"),
    cq(101, "stability ambiguous across domains"),
    cq(48, "attachment ambiguous: flow vs structural"),
    cq(40, "transition in hypersonic wakes: wake vs other meanings"),
    cq(110, "buckling mode: structural vs instability mode"),
    cq(56, "steady-state aerodynamic data for flutter: flutter vs other vibration"),
    cq(57, "steady and non-steady flow characteristics for flutter: flow vs structural"),
    cq(209, "separated: boundary-layer separation vs structural separation"),
    cq(38, "transition in hypersonic wake: body geometry ambiguity"),
    cq(42, "criterion for transonic flow: linearized theory validity, multiple senses"),
]
ambig_custom = [
    custom("ambig_c1", "stability analysis of thin shell structures",
           "structural buckling (NOT flow stability)", [],
           "stability: structural buckling vs flow stability"),
    custom("ambig_c2", "stability of laminar flow on heated surfaces",
           "flow stability (NOT structural)", ["73","55","608"],
           "stability: flow meaning, not structural"),
    custom("ambig_c3", "performance of supersonic compressor stages",
           "axial compressor (NOT data compression)", ["588","589","590"],
           "compressor: mechanical vs data"),
    custom("ambig_c4", "effects of body shape on drag in high speed flight",
           "aerodynamic drag (NOT rigid body dynamics)", ["77","25","20","27"],
           "body: aerodynamic body vs rigid body"),
    custom("ambig_c5", "recovery of pressure behind normal shock",
           "pressure recovery (NOT thermal recovery factor)", ["64","65"],
           "recovery: pressure vs thermal"),
    custom("ambig_c6", "deflection of panels under combined loading",
           "structural deflection (NOT flow deflection)", [],
           "deflection: structural vs flow turning"),
    custom("ambig_c7", "separation point prediction for airfoil flows",
           "flow separation (NOT physical separation of parts)", ["54","55","458","459"],
           "separation: flow vs physical"),
    custom("ambig_c8", "Reynolds number effects on model testing",
           "fluid dynamics Re effects (NOT Reynolds stress modeling)", ["7","8","9","40","41"],
           "Reynolds number: multiple contexts"),
    custom("ambig_c9", "flutter boundary determination for wing structures",
           "aeroelastic flutter (NOT acoustic flutter)", ["14","15","52","391"],
           "flutter: aeroelastic vs acoustic"),
    custom("ambig_c10", "interaction between shock and boundary layer",
           "SWBLI (NOT shock-shock interaction)", ["794","797","798"],
           "interaction: shock-BL vs shock-shock"),
]

# ============================================================
# 5. OUT-OF-VOCABULARY
# ============================================================
oov_cran = [
    cq(44, "chapman-enskog: rare proper noun"),
    cq(20, "joule heating + magnetohydrodynamic: variant spellings"),
    cq(13, "aileron buzz: rare compound term"),
    cq(170, "pitot-static tubes + orthodox: unusual terms"),
    cq(173, "Lyapunov method: proper noun OOV"),
    cq(172, "blasius problem: proper noun"),
    cq(176, "Biot principle: proper noun"),
    cq(153, "navier-stokes difference equations: compound proper name"),
    cq(154, "elliptic difference equations: specialized math term"),
    cq(49, "blasius equation + three-point BVP: combined rare terms"),
    cq(112, "clamped plate + membrane boundary value: specialized structural terms"),
]
oov_custom = [
    custom("oov_c1", "MHD effects on convective heat transfer in conducting fluids",
           "magnetohydrodynamic free convection", ["87","88","61"],
           "MHD abbreviation not in 1960s corpus"),
    custom("oov_c2", "CFD solutions for flow over bluff bodies",
           "computational fluid dynamics", [],
           "CFD abbreviation absent from corpus"),
    custom("oov_c3", "RANS turbulence modeling for compressible boundary layers",
           "Reynolds-Averaged turbulence modeling", ["16","17"],
           "RANS abbreviation absent"),
    custom("oov_c4", "TPS material performance during ballistic re-entry",
           "thermal protection system ablation", ["82"],
           "TPS abbreviation OOV"),
    custom("oov_c5", "aeroelastic divergence speed of cantilever wings",
           "aeroelastic divergence", ["12","14","15"],
           "divergence speed may not appear verbatim"),
    custom("oov_c6", "Tollmien-Schlichting wave amplification in boundary layers",
           "BL instability waves", ["73"],
           "Tollmien-Schlichting: specialized proper noun"),
    custom("oov_c7", "Karman vortex street behind bluff cylinders",
           "vortex shedding behind cylinders", [],
           "Karman vortex street: compound proper name"),
    custom("oov_c8", "Prandtl-Meyer expansion fan at supersonic corners",
           "expansion waves at corners", [],
           "Prandtl-Meyer: compound proper noun"),
    custom("oov_c9", "LEO satellite orbital decay due to atmospheric drag",
           "satellite orbit contraction", ["613","614","615","548"],
           "LEO abbreviation + modern terminology"),
    custom("oov_c10", "FEM analysis of thin-walled aerospace structures",
           "finite element structural analysis", [],
           "FEM abbreviation not in corpus"),
    custom("oov_c11", "RCS thruster effects on vehicle attitude control",
           "reaction control system", [],
           "RCS abbreviation OOV"),
]

# ============================================================
# 6. LACK OF CONTEXTUAL REPRESENTATION
# ============================================================
ctx_cran = [
    cq(90, "shock-induced boundary-layer separation: 4-word compound phrase"),
    cq(9, "slip flow heat transfer: multi-word concept"),
    cq(164, "onset of shock-induced BL separation: compound phrase"),
    cq(125, "jet interference with supersonic flow: compound concept"),
    cq(126, "thrust vector control by fluid injection: multi-word phrases"),
    cq(181, "jet interference with supersonic flows theoretical: similar to 125"),
    cq(14, "shock-sound wave interaction: compound concept"),
    cq(30, "flow visualization on slender conical wings: phrase"),
    cq(109, "panels subjected to aerodynamic heating: phrase context matters"),
    cq(71, "experimental results on hypersonic viscous interaction: phrase"),
    cq(192, "uniformly loaded sectors: phrase-level meaning"),
]
ctx_custom = [
    custom("ctx_c1", "shock wave boundary layer interaction in supersonic flow",
           "SWBLI compound phrase", ["794","797","798","793","796"],
           "compound phrase fragmented by BoW"),
    custom("ctx_c2", "free stream turbulence effects on transition",
           "how freestream turbulence affects transition", ["40","41","7","8","608"],
           "free stream turbulence is a phrase"),
    custom("ctx_c3", "heat transfer at stagnation point on blunt nose",
           "stagnation-point heat transfer", ["24","283","35","36"],
           "multi-word technical phrase"),
    custom("ctx_c4", "wind tunnel wall interference corrections",
           "wall interference in wind tunnels", ["252","672","431"],
           "4-word specific concept"),
    custom("ctx_c5", "zero angle of attack pressure distribution on ogive cylinder",
           "pressure at zero incidence on ogive", ["56","57"],
           "phrases fragment into common words"),
    custom("ctx_c6", "laminar to turbulent boundary layer transition mechanism",
           "BL transition mechanism", ["7","8","9","40","41","79"],
           "multi-word phrase: laminar-turbulent transition"),
    custom("ctx_c7", "flat plate skin friction coefficient in compressible flow",
           "compressible skin friction on flat plate", ["50","49","9"],
           "flat plate skin friction: 4-word phrase"),
    custom("ctx_c8", "circular cylindrical shell buckling under axial compression",
           "axial compression buckling of cylinders", ["821","822","824","760"],
           "compound phrase loses meaning in BoW"),
    custom("ctx_c9", "blunt body inviscid flow field at hypersonic speed",
           "hypersonic blunt body flow", ["25","35","36","20"],
           "word order and proximity matter"),
    custom("ctx_c10", "panel flutter at high supersonic mach numbers",
           "supersonic panel flutter", ["15","52","858","859"],
           "phrase: panel flutter at supersonic, not general flutter"),
    custom("ctx_c11", "creep buckling of columns under sustained load",
           "column creep buckling", ["1018","1020","1014","950"],
           "creep buckling: compound concept"),
]

# ============================================================
# 7. SPARSE REPRESENTATIONS
# ============================================================
sparse_cran = [
    cq(93, "methane-air combustion: very few shared terms"),
    cq(15, "photoelastic materials: rare term, sparse vector"),
    cq(78, "kink in surge line of axial compressor: highly specific"),
    cq(128, "pump design for digital computer: specialized terms"),
    cq(216, "wave system from pressure distribution over liquid surface: niche"),
    cq(103, "buckling of sandwich sphere: rare combination"),
    cq(105, "internal liquid column on breathing vibrations of shell: niche"),
    cq(31, "end plate simulating 2D flow over bluff cylinder: specific"),
    cq(12, "aerodynamic performance of channel flow ground effect machines: niche"),
    cq(129, "simplified pump design equation from 3D incompressible flow: specialized"),
]
sparse_custom = [
    custom("sparse_c1", "photoelastic stress analysis techniques for wing structures",
           "photothermoelastic measurement", ["463","462","30"],
           "photoelastic is rare, sparse TF-IDF vector"),
    custom("sparse_c2", "magneto-gasdynamic acceleration in constant-area channels",
           "MHD flow acceleration", ["33","34"],
           "magneto-gasdynamic appears in few docs"),
    custom("sparse_c3", "creep rupture time prediction for pressurized cylindrical vessels",
           "creep failure of cylinders", [],
           "creep rupture: sparse overlap"),
    custom("sparse_c4", "sandwich shell buckling under combined thermal and mechanical loading",
           "sandwich structure buckling", [],
           "specialized structural query, sparse terms"),
    custom("sparse_c5", "gyroscopic precession effects on propeller-engine vibration",
           "propeller gyroscopic effects", ["42","78"],
           "gyroscopic precession: very rare terms"),
    custom("sparse_c6", "transpiration cooling effectiveness on hemisphere nose",
           "transpiration-cooled hemisphere", ["662","628"],
           "transpiration cooling: niche topic"),
    custom("sparse_c7", "acoustic fatigue of skin panels in jet noise environment",
           "acoustic fatigue of panels", ["75","724","726","727","640"],
           "acoustic fatigue: specialized compound term"),
    custom("sparse_c8", "torispherical head buckling under internal pressure",
           "torispherical shell buckling", ["1134","1137","1138","1071"],
           "torispherical: rare geometric term"),
    custom("sparse_c9", "Couette flow behavior between rotating cylinders",
           "turbulent Couette flow", ["99"],
           "Couette: specialized term, sparse vector"),
    custom("sparse_c10", "corrugated core sandwich cylinder stability equations",
           "sandwich cylinder buckling", ["1046","1048","1050","956"],
           "corrugated core sandwich: very specific terms"),
]

# ============================================================
# BUILD FINAL JSON
# ============================================================
test_set = {
    "description": "Expanded test set: 20+ queries per limitation. Set A = Cranfield queries (verified against cran_qrels.json). Set B = custom queries.",
    "limitations": {
        "lack_of_semantic_understanding": {
            "abbreviation": "Sem.",
            "methods_that_address": ["WordNet expansion","Word-embedding expansion","LSA","ESA","WordNet local context"],
            "cranfield_queries": sem_cran,
            "custom_queries": sem_custom,
        },
        "high_dimensionality_and_computational_cost": {
            "abbreviation": "Dim./Cost",
            "methods_that_address": ["LSA","Query reduction"],
            "cranfield_queries": dim_cran,
            "custom_queries": dim_custom,
        },
        "poor_scalability": {
            "abbreviation": "Scale",
            "methods_that_address": ["LSA (4/10)","ESA (4/10)"],
            "cranfield_queries": scale_cran,
            "custom_queries": scale_custom,
        },
        "word_sense_ambiguity": {
            "abbreviation": "Ambig.",
            "methods_that_address": ["n-gram","Unordered local-context BoW","WordNet (4/10)","LSA (4/10)","ESA (4/10)","Query reduction (4/10)","WordNet local context"],
            "cranfield_queries": ambig_cran,
            "custom_queries": ambig_custom,
        },
        "out_of_vocabulary": {
            "abbreviation": "OOV",
            "methods_that_address": ["WordNet expansion (X)","Word-embedding expansion (5/10)","ESA (5/10)","WordNet local context (X)"],
            "cranfield_queries": oov_cran,
            "custom_queries": oov_custom,
        },
        "lack_of_contextual_representation": {
            "abbreviation": "Context",
            "methods_that_address": ["n-gram (X)","Unordered local-context BoW (X)","WordNet local context (X)"],
            "cranfield_queries": ctx_cran,
            "custom_queries": ctx_custom,
        },
        "sparse_representations": {
            "abbreviation": "Sparse",
            "methods_that_address": ["LSA (X)","ESA (X)"],
            "cranfield_queries": sparse_cran,
            "custom_queries": sparse_custom,
        },
    }
}

# ============================================================
# VERIFICATION
# ============================================================
print("=" * 70)
print("GENERATING & VERIFYING EXPANDED TEST SET")
print("=" * 70)

total_cran = 0
total_cust = 0
errors = []

for lim_key, lim in test_set["limitations"].items():
    cqs = lim["cranfield_queries"]
    cuqs = lim["custom_queries"]
    n = len(cqs) + len(cuqs)
    total_cran += len(cqs)
    total_cust += len(cuqs)

    status = "OK" if n >= 20 else f"BELOW 20 ({n})"
    print(f"\n  {lim_key}: {len(cqs)} cranfield + {len(cuqs)} custom = {n} [{status}]")

    for c in cqs:
        qn = c["query_number"]
        if qn not in qmap:
            errors.append(f"[ERROR] {lim_key}: Q{qn} not in cran_queries.json")
        for did in c["relevant_docs"]:
            if did not in set(e["id"] for e in qrels.get(qn, [])):
                errors.append(f"[ERROR] {lim_key}: Q{qn} doc {did} not in qrels")

print(f"\n{'='*70}")
print(f"TOTALS: {total_cran} Cranfield + {total_cust} Custom = {total_cran+total_cust} queries")
print(f"Errors: {len(errors)}")
if errors:
    for e in errors:
        print(f"  {e}")
else:
    print("*** ALL CRANFIELD QUERIES AND DOC IDS VERIFIED ***")

# Save
out_path = os.path.join(BASE, "project", "limitation_test_set_expanded.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(test_set, f, indent=2, ensure_ascii=False)
print(f"\nSaved to: {out_path}")

# Print summary table
print(f"\n{'='*70}")
print(f"{'Limitation':<45} {'Cranfield':>10} {'Custom':>10} {'Total':>10}")
print(f"{'-'*45} {'-'*10} {'-'*10} {'-'*10}")
for lim_key, lim in test_set["limitations"].items():
    nc = len(lim["cranfield_queries"])
    nu = len(lim["custom_queries"])
    print(f"{lim_key:<45} {nc:>10} {nu:>10} {nc+nu:>10}")
print(f"{'-'*45} {'-'*10} {'-'*10} {'-'*10}")
print(f"{'TOTAL':<45} {total_cran:>10} {total_cust:>10} {total_cran+total_cust:>10}")
