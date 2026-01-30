import os
import zipfile
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import json

class DataExtractor:
    def __init__(self, zip_path, output_dir="extracted_data"):
        self.zip_path = zip_path
        self.output_dir = output_dir
        self.subjects_data = []
        
    def extract_zip(self):
        """Extrait le zip contenant tous les sujets"""
        print("📦 Extraction du fichier ZIP...")
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)
        print(f"✅ Extraction terminée dans {self.output_dir}")
    
    def parse_fastq(self, fastq_path, max_sequences=1000):
        """
        Parse un fichier FASTQ et extrait les séquences
        """
        sequences = []
        try:
            with open(fastq_path, 'r') as handle:
                for i, record in enumerate(SeqIO.parse(handle, "fastq")):
                    if i >= max_sequences:
                        break
                    sequences.append({
                        'id': record.id,
                        'sequence': str(record.seq),
                        'quality': record.letter_annotations.get("phred_quality", [])
                    })
        except Exception as e:
            print(f"⚠️ Erreur lors du parsing de {fastq_path}: {e}")
        return sequences
    
    def find_subject_folders(self, root_dir):
        """
        ✅ CORRECTION: Trouve tous les dossiers Subject_* récursivement
        """
        subject_folders = []
        root_path = Path(root_dir)
        
        # Cherche récursivement tous les dossiers commençant par "Subject_"
        for item in root_path.rglob("Subject_*"):
            if item.is_dir():
                subject_folders.append(item)
        
        return subject_folders
    
    def process_all_subjects(self):
        """
        ✅ CORRECTION: Parcourt tous les dossiers de sujets (même imbriqués)
        """
        print("\n🔍 Traitement de tous les sujets...")
        
        # Trouve tous les dossiers Subject_*
        subject_folders = self.find_subject_folders(self.output_dir)
        
        if len(subject_folders) == 0:
            print("❌ Aucun dossier Subject_* trouvé!")
            print(f"   Vérifiez la structure dans {self.output_dir}")
            
            # Debug: affiche la structure trouvée
            print("\n📂 Structure détectée:")
            for item in Path(self.output_dir).iterdir():
                print(f"   {item}")
                if item.is_dir():
                    for subitem in item.iterdir():
                        print(f"      → {subitem.name}")
            
            return []
        
        print(f"✅ {len(subject_folders)} dossiers de sujets trouvés")
        
        for subject_folder in subject_folders:
            subject_id = subject_folder.name
            print(f"\n📊 Traitement: {subject_id}")
            
            # Lecture du fichier clinical.csv
            clinical_path = subject_folder / "clinical.csv"
            clinical_data = {}
            
            if clinical_path.exists():
                try:
                    df = pd.read_csv(clinical_path)
                    if len(df) > 0:
                        clinical_data = df.iloc[0].to_dict()
                        print(f"  ✅ Données cliniques chargées")
                except Exception as e:
                    print(f"  ⚠️ Erreur lecture clinical.csv: {e}")
            else:
                print(f"  ⚠️ clinical.csv non trouvé")
            
            # Lecture des fichiers FASTQ
            fastq_dir = subject_folder / "fastq"
            fastq_files = []
            sequences_data = []
            
            if fastq_dir.exists():
                # Cherche tous les fichiers .fastq et .fastq.gz
                fastq_files = list(fastq_dir.glob("*.fastq*"))
                print(f"  📄 {len(fastq_files)} fichiers FASTQ trouvés")
                
                # Parse les 2 premiers fichiers
                for fastq_file in fastq_files[:2]:
                    print(f"    → Parsing: {fastq_file.name}")
                    seqs = self.parse_fastq(fastq_file, max_sequences=500)
                    sequences_data.extend(seqs)
                    print(f"       {len(seqs)} séquences extraites")
            else:
                print(f"  ⚠️ Dossier fastq/ non trouvé")
            
            # Stockage des données du sujet
            subject_info = {
                'subject_id': subject_id,
                'clinical_data': clinical_data,
                'sequences': sequences_data,
                'num_sequences': len(sequences_data),
                'fastq_files': [f.name for f in fastq_files]
            }
            
            self.subjects_data.append(subject_info)
            print(f"  ✅ Total: {len(sequences_data)} séquences extraites")
        
        print(f"\n🎉 Traitement terminé: {len(self.subjects_data)} sujets")
        return self.subjects_data
    
    def save_processed_data(self, output_file="processed_subjects.json"):
        """Sauvegarde les données traitées en JSON"""
        print(f"\n💾 Sauvegarde des données dans {output_file}...")
        
        # Limite les séquences sauvegardées
        data_to_save = []
        for subject in self.subjects_data:
            limited_subject = subject.copy()
            # Garde seulement les 100 premières séquences pour économiser l'espace
            limited_subject['sequences'] = subject['sequences'][:100]
            data_to_save.append(limited_subject)
        
        with open(output_file, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"✅ Données sauvegardées!")
        
        # Statistiques
        total_sequences = sum(s['num_sequences'] for s in self.subjects_data)
        avg_sequences = total_sequences / len(self.subjects_data) if len(self.subjects_data) > 0 else 0
        
        print(f"\n📊 Statistiques:")
        print(f"  • Nombre de sujets: {len(self.subjects_data)}")
        print(f"  • Séquences totales: {total_sequences}")
        print(f"  • Moyenne par sujet: {avg_sequences:.0f}")
        
        # Affiche quelques exemples
        if len(self.subjects_data) > 0:
            print(f"\n📋 Exemples de sujets traités:")
            for subject in self.subjects_data[:3]:
                print(f"  • {subject['subject_id']}: {subject['num_sequences']} séquences")

# Utilisation
if __name__ == "__main__":
    extractor = DataExtractor("data/subjectid.zip")
    
    # Extraction (commentez si déjà fait)
    extractor.extract_zip()
    
    # Traitement
    subjects_data = extractor.process_all_subjects()
    
    # Sauvegarde
    if len(subjects_data) > 0:
        extractor.save_processed_data()
    else:
        print("\n❌ Aucun sujet traité - vérifiez la structure du ZIP")