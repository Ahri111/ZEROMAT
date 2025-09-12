import re
import os
from openai import OpenAI

print("✅ Libraries imported successfully!")

# ========================================
# Cell 2: Set API key
# ========================================
# Replace with your actual API key
API_KEY = "your-actual-api-key-here"  # Replace with your real key

print("✅ API key configured!")

# ========================================
# Cell 3: Define BandgapFeatureRecommender class
# ========================================
class BandgapFeatureRecommender:
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
    
    def display_results(self, query, ai_response, features):
        """Display results nicely in Jupyter notebook"""
        print(f"Query: {query}")
        print("=" * 80)
        print(f"AI Response:\n{ai_response}")
        print("=" * 80)
        print("Parsed Features:")
        for feature in features:
            print(f"- {feature}")
        print(f"\nExtracted features for MP data fetcher: {features}")
        return features
    
    def ask(self, query=None):
        """Single question function - perfect for notebooks"""
        if query is None:
            query = input("Enter your question: ")
        
        print("Getting recommendations...")
        
        # Get AI response
        ai_response = self.get_recommendations(query)
        
        # Extract features
        features = self.parse_features(ai_response)
        
        # Display results
        return self.display_results(query, ai_response, features)
    
    def interactive_ask(self):
        """Interactive asking mode"""
        while True:
            query = input("\nEnter your question (type 'quit' to exit): ")
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query.strip():
                print("Please enter a question!")
                continue
                
            print("Getting recommendations...")
            
            # Get AI response
            ai_response = self.get_recommendations(query)
            
            # Extract features
            features = self.parse_features(ai_response)
            
            # Display results
            self.display_results(query, ai_response, features)
            print("\n" + "="*60)

print("✅ BandgapFeatureRecommender class defined successfully!")

# ========================================
# Cell 4: Initialize the recommender system
# ========================================
recommender = BandgapFeatureRecommender()

print("✅ Bandgap Feature Recommender initialized!")

# ========================================
# Cell 5: Define quick test function
# ========================================
def quick_test():
    """Quick test with a simple question"""
    print("Quick Test Mode")
    print("-" * 30)
    
    test_query = "What are the most important features for predicting bandgap?"
    print(f"Test question: {test_query}")
    
    return recommender.ask(test_query)

print("✅ All functions defined successfully!")
print("\n" + "="*60)
print("🚀 Ready to use!")
print("="*60)
print("\nTry in the next cell:")
print("1. quick_test()  # Quick test")
print("2. recommender.ask()  # Interactive input")
print("3. recommender.ask('your question here')  # Direct question")
print("4. recommender.interactive_ask()  # Multiple questions")
print("\nExample:")
print("recommender.ask('What features predict semiconductor bandgaps?')")
