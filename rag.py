# Clean Bandgap Feature Recommender
# No errors, simple and direct

import re
from openai import OpenAI

# PUT YOUR API KEY HERE - ONLY PLACE TO CHANGE
API_KEY = "your own api"
    """Clean feature recommender for bandgap prediction"""
    
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)
        
        # All available features from your MP data fetcher
        self.features = {
            "density": "Material density in g/cm³",
            "volume": "Unit cell volume in Ų", 
            "nsites": "Number of sites in unit cell",
            "space_group": "Space group symmetry information",
            "crystal_system": "Crystal system classification",
            "formation_energy_per_atom": "Formation energy per atom in eV",
            "energy_above_hull": "Energy above convex hull in eV",
            "uncorrected_energy_per_atom": "Raw DFT energy per atom",
            "band_gap": "Electronic band gap in eV",
            "is_metal": "Boolean metallic character",
            "is_magnetic": "Boolean magnetic ordering",
            "total_magnetization": "Total magnetic moment",
            "num_elements": "Number of different elements",
            "composition": "Complete elemental composition",
            "elements": "List of constituent elements",
            "chemsys": "Chemical system notation",
            "formula_pretty": "Pretty chemical formula",
            "bulk_modulus": "Bulk modulus in GPa",
            "shear_modulus": "Shear modulus in GPa",
            "universal_anisotropy": "Elastic anisotropy measure",
            "dielectric_total": "Total dielectric constant",
            "dielectric_ionic": "Ionic dielectric contribution",
            "dielectric_electronic": "Electronic dielectric contribution"
        }
    
    def get_recommendations(self, query):
        """Get feature recommendations from AI"""
        
        features_text = "\n".join([f"- {name}: {desc}" for name, desc in self.features.items()])
        
        prompt = f"""You are a materials scientist expert in bandgap prediction.

User question: "{query}"

Available features:
{features_text}

Based on physics and machine learning principles, recommend the most relevant features.

Important domain knowledge:
- Density and formation energy are usually most predictive
- Structural features (volume, crystal system) are very important
- Mechanical properties can provide additional insights
- Electronic properties like is_metal are obvious indicators

Format your response EXACTLY like this:
**RECOMMENDED FEATURES:**
1. feature_name - physics-based reason
2. feature_name - physics-based reason
3. feature_name - physics-based reason
4. feature_name - physics-based reason (if relevant)
5. feature_name - physics-based reason (if relevant)
... (continue numbering for all relevant features)
Only use feature names from the list above."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API Error: {str(e)}"
    
    def parse_features(self, ai_response):
        """Extract feature names from AI response"""
        pattern = r'\d+\.\s*([a-zA-Z_]+)'
        matches = re.findall(pattern, ai_response)
        
        # Only keep valid features
        valid_features = [f for f in matches if f in self.features]
        return valid_features
    
    def save_results(self, query, ai_response, features):
        """Save results to file for MP data fetcher"""
        filename = "feature_recommendations.txt"
        
        with open(filename, "w") as f:
            f.write(f"Query: {query}\n")
            f.write("=" * 80 + "\n")
            f.write(f"AI Response:\n{ai_response}\n")
            f.write("=" * 80 + "\n")
            f.write("Parsed Features:\n")
            for feature in features:
                f.write(f"- {feature}\n")
        
        print(f"\nSaved to: {filename}")
        print(f"Extracted features: {features}")
        return features
    
    def chat(self):
        """Interactive chat mode"""
        print("Type 'quit' to exit\n")
        
        while True:
            query = input("Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("\nGetting recommendations...")
            
            # Get AI response
            ai_response = self.get_recommendations(query)
            
            print("\n" + "=" * 60)
            print("AI RESPONSE:")
            print("=" * 60)
            print(ai_response)
            
            # Extract features
            features = self.parse_features(ai_response)
            
            # Save results
            self.save_results(query, ai_response, features)
            
            print(f"\nRecommended for MP data fetcher: {features}")
            print("\n" + "=" * 60)

def quick_test():
    """Quick test with a simple question"""
    print("Quick Test Mode")
    print("-" * 30)
    
    recommender = BandgapFeatureRecommender()
    
    test_query = "What are the most important features for predicting bandgap?"
    print(f"Test question: {test_query}")
    
    ai_response = recommender.get_recommendations(test_query)
    features = recommender.parse_features(ai_response)
    
    print(f"\nAI Response:\n{ai_response}")
    print(f"\nExtracted features: {features}")
    
    recommender.save_results(test_query, ai_response, features)

def main():
    """Main function"""
    print("Choose mode:")
    print("1. Interactive chat")
    print("2. Quick test")
    
    choice = input("Enter 1 or 2: ").strip()
    
    recommender = BandgapFeatureRecommender()
    
    if choice == "1":
        recommender.chat()
    else:
        quick_test()

if __name__ == "__main__":
    main()