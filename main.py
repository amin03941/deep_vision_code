"""
🧬 PIPELINE COMPLET: Genetic Neighbor Finder + Evaluation + Therapeutic Analysis
"""

import sys
from pathlib import Path
import config

def step_1_extract():
    print("\n" + "="*70)
    print("ÉTAPE 1: EXTRACTION ET PARSING")
    print("="*70)
    
    from scripts.extract_and_parse import DataExtractor
    
    extractor = DataExtractor(
        config.ZIP_PATH,
        output_dir=config.EXTRACTED_DATA_DIR
    )
    
    if not Path(config.PROCESSED_SUBJECTS_FILE).exists():
        if not Path(config.EXTRACTED_DATA_DIR).exists():
            extractor.extract_zip()
        extractor.process_all_subjects()
        extractor.save_processed_data(config.PROCESSED_SUBJECTS_FILE)
    else:
        print(f"✅ Données déjà extraites")

def step_2_vectorize():
    print("\n" + "="*70)
    print("ÉTAPE 2: VECTORISATION MULTIMODALE (CORRIGÉE)")
    print("="*70)
    
    import json
    from scripts.vectorize import MultimodalVectorizer
    
    # ✅ Force la re-vectorisation si vectors.pkl existe mais est ancien
    if Path(config.VECTORS_FILE).exists():
        print("⚠️ vectors.pkl existe déjà")
        response = input("Voulez-vous re-vectoriser avec la nouvelle méthode? (o/n): ")
        if response.lower() != 'o':
            print("✅ Vectorisation existante conservée")
            return
    
    with open(config.PROCESSED_SUBJECTS_FILE, 'r') as f:
        subjects_data = json.load(f)
    
    vectorizer = MultimodalVectorizer(
        k=config.K_MER_SIZE,
        genetic_dim=config.GENETIC_VECTOR_DIM
    )
    
    vectorizer.vectorize_all_subjects(
        subjects_data,
        alpha=config.ALPHA_GENETIC,
        output_file=config.VECTORS_FILE
    )

def step_3_index():
    print("\n" + "="*70)
    print("ÉTAPE 3: INDEXATION QDRANT")
    print("="*70)
    
    import pickle
    from scripts.index_qdrant import QdrantIndexer
    
    with open(config.VECTORS_FILE, 'rb') as f:
        data = pickle.load(f)
        vectors_data = data['vectors']
    
    indexer = QdrantIndexer(
        collection_name=config.QDRANT_COLLECTION_NAME,
        path=config.QDRANT_DATA_DIR
    )
    
    # Re-index si vectors.pkl a été régénéré
    indexer.index_vectors(vectors_data)

def step_4_evaluate():
    """
    ✅ NOUVEAU: Évaluation du système avant analyse
    """
    print("\n" + "="*70)
    print("ÉTAPE 4: ÉVALUATION DU SYSTÈME")
    print("="*70)
    
    from scripts.evaluate import SystemEvaluator
    
    evaluator = SystemEvaluator(
        qdrant_path=config.QDRANT_DATA_DIR,
        collection_name=config.QDRANT_COLLECTION_NAME
    )
    
    # Évaluation avec k par défaut
    result = evaluator.evaluate_leave_one_out(top_k=config.TOP_K_NEIGHBORS)
    
    # Test de différentes valeurs de k (optionnel)
    print("\n" + "="*70)
    response = input("Voulez-vous tester différentes valeurs de k? (o/n): ")
    if response.lower() == 'o':
        evaluator.test_different_k_values(config.K_VALUES_TO_TEST)
    
    return result

def step_5_analyze(subject_id):
    print(f"\n{'='*60}")
    print(f"ÉTAPE 5: ANALYSE DU PATIENT {subject_id}")
    print(f"{'='*60}")
    
    from scripts.search_neighbors import NeighborAnalyzer
    from scripts.therapeutic_analysis import TherapeuticAnalyzer
    
    analyzer = NeighborAnalyzer(
        qdrant_path=config.QDRANT_DATA_DIR,
        collection_name=config.QDRANT_COLLECTION_NAME
    )
    
    analysis = analyzer.analyze_patient(subject_id, top_k=config.TOP_K_NEIGHBORS)
    
    if not analysis:
        print(f"❌ Impossible d'analyser {subject_id}")
        return None
    
    # Génération rapport thérapeutique
    if config.GEMINI_API_KEY and len(config.GEMINI_API_KEY) > 20:
        therapeutic_analyzer = TherapeuticAnalyzer(
            api_key=config.GEMINI_API_KEY
        )
        report = therapeutic_analyzer.generate_therapeutic_report(analysis)
        
        if report:
            print("\n" + "="*70)
            print("📄 RAPPORT THÉRAPEUTIQUE GÉNÉRÉ")
            print("="*70)
            print(report)
    else:
        print("\n⚠️ Clé API Gemini non configurée")
        print("   Analyse des voisins terminée, mais rapport non généré")
    
    return analysis

def interactive_mode():
    print("\n" + "="*70)
    print("🎯 MODE INTERACTIF")
    print("="*70)
    
    import pickle
    
    with open(config.VECTORS_FILE, 'rb') as f:
        data = pickle.load(f)
        vectors_data = data['vectors']
    
    print(f"\n📊 {len(vectors_data)} sujets disponibles")
    print("\nExemples de SubjectID:")
    
    # Montre des exemples de chaque classe
    for cls in config.PATIENT_CLASSES:
        examples = [v for v in vectors_data if v['class'] == cls][:2]
        for v in examples:
            print(f"  - {v['subject_id']} (Classe: {v['class']})")
    
    print("\n💡 Tapez un SubjectID pour l'analyser")
    print("   (vous pouvez omettre 'Subject_' au début)")
    print("   Tapez 'list' pour voir tous les sujets")
    print("   Tapez 'quit' pour quitter")
    
    while True:
        print("\n" + "-"*70)
        subject_id = input("\nSubjectID (ou 'quit'/'list'): ").strip()
        
        if subject_id.lower() in ['quit', 'exit', 'q', '']:
            break
        
        if subject_id.lower() == 'list':
            print("\n📋 Liste complète des sujets:")
            for v in vectors_data:
                print(f"  {v['subject_id']} - {v['class']}")
            continue
        
        if not subject_id.startswith('Subject_'):
            subject_id = 'Subject_' + subject_id
        
        step_5_analyze(subject_id)

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🧬 GENETIC NEIGHBOR FINDER (Version Corrigée) 🧬           ║
    ║                                                              ║
    ║  Pipeline avec:                                             ║
    ║  ✅ Vectorisation k-mer hashing (vecteurs comparables)      ║
    ║  ✅ Fusion multimodale pondérée                             ║
    ║  ✅ Évaluation Leave-One-Out                                ║
    ║  ✅ Rapports thérapeutiques réalistes                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    config.get_config_summary()
    
    if not config.validate_config():
        print("\n❌ Configuration incomplète - vérifiez config.py")
        sys.exit(1)
    
    try:
        # Pipeline complet
        step_1_extract()
        step_2_vectorize()
        step_3_index()
        
        # ✅ NOUVEAU: Évaluation obligatoire
        eval_result = step_4_evaluate()
        
        if eval_result['accuracy'] < config.ACCURACY_THRESHOLD_MEDIUM:
            print("\n⚠️ ATTENTION: Accuracy faible (<50%)")
            print("   Le système de voisinage n'est pas fiable")
            response = input("Voulez-vous continuer quand même? (o/n): ")
            if response.lower() != 'o':
                print("\n👋 Arrêt du programme")
                return
        
        # Mode interactif
        interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt du programme")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()