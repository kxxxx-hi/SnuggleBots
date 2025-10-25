from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from .synonyms import SYNONYMS, canonicalize
from .responses import get_response

from rapidfuzz import process
from transformers.utils import logging as hf_logging
import warnings, re

# Silence noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

# Required and optional entities for pet search
REQUIRED_ENTITIES = ["PET_TYPE", "STATE"]
OPTIONAL_ENTITIES = ["BREED", "COLOR", "SIZE", "GENDER", "AGE", "FURLENGTH"]

# ---------------------------------------------------------------------------
# AUTOCORRECT
# ---------------------------------------------------------------------------
def autocorrect_text(text, known_words=None, threshold=80):
    """Light typo correction for meaningful words only."""
    if known_words is None:
        known_words = [
            "dog", "cat", "adopt", "adoption", "puppy", "kitten",
            "male", "female", "small", "large", "brown", "white", "black",
            "golden", "cream", "short", "long", "fur",
            "Johor", "Penang", "Melaka", "Selangor", "Kuala", "Lumpur",
            "Perak", "Sabah", "Sarawak"
        ]
    words, corrected = text.split(), []
    for w in words:
        if len(w) <= 2:  # skip very short tokens
            corrected.append(w)
            continue
        match, score, _ = process.extractOne(w, known_words)
        corrected.append(match if score >= threshold and match.lower() != w.lower() else w)
    return " ".join(corrected)

# ---------------------------------------------------------------------------
# CHATBOT PIPELINE
# ---------------------------------------------------------------------------
class ChatbotPipeline:
    """End-to-end pet adoption chatbot with clean fallbacks and context."""

    def __init__(self, rag_system=None, pet_search_func=None, azure_components=None):
        self.intent_clf = IntentClassifier()
        self.ner_extractor = EntityExtractor()
        self.rag_system = rag_system  # Will be None for now
        self.pet_search_func = pet_search_func  # Function to perform pet search
        self.azure_components = azure_components  # Azure components for pet search
        self.session = {"intent": None, "entities": {}, "greeted": False}

    # -----------------------------------------------------------------------
    # MAIN MESSAGE HANDLER
    # -----------------------------------------------------------------------
    def handle_message(self, user_input: str) -> str:
        user_input = user_input.strip()

        # --- Show greeting if user presses Enter at start ---
        if not user_input:
            return self._get_greeting()

        # --- Typo correction ---
        user_input = autocorrect_text(user_input)

        # --- Small talk shortcuts ---
        lower = user_input.lower()
        if lower in ["no", "nope", "nah"]:
            return "Alright 😊 Let me know anytime if you change your mind."
        if lower in ["hi", "hey", "hello"]:
            self.session["greeted"] = True
            return get_response("greeting")

        # --- Intent classification ---
        intent, conf = self.intent_clf.predict(user_input)

        # --- Fix false greeting classification ---
        greeting_words = ["hi", "hey", "hello"]
        if intent == "greeting":
            if lower not in greeting_words and len(user_input.split()) <= 2 and not any(
                g in lower for g in greeting_words
            ):
                intent = "unknown"

        # --- Maintain find_pet context for short follow-ups (color/size/gender/state) ---
        if self.session.get("intent") == "find_pet":
            if len(user_input.split()) <= 2 and len(user_input) > 2 and intent in ["unknown", "other"]:
                intent = "find_pet"

        # --- Initial greeting check ---
        if not self.session["greeted"]:
            self.session["greeted"] = True
            if intent == "greeting":
                return self._get_greeting()

        prev_intent = self.session.get("intent")

        # --- Handle unknown / low-confidence intents gracefully ---
        if intent == "unknown" or conf < 0.55:
            return self._handle_unknown(user_input)

        # --- Intent switching ---
        if self._is_new_intent(intent, prev_intent):
            self.session = {"intent": intent, "entities": {}, "greeted": True}

        self.session["intent"] = intent

        # --- Route by intent ---
        if intent == "find_pet":
            return self._handle_find_pet(user_input)
        if intent == "pet_care":
            return self._handle_pet_care(user_input)
        if intent == "thank_you":
            return get_response("thank_you")
        if intent == "greeting":
            return get_response("greeting")
        if intent == "goodbye":
            return get_response("goodbye")

        # --- Default fallback ---
        return get_response("unknown")

    # -----------------------------------------------------------------------
    # PET CARE HANDLER (RAG Integration)
    # -----------------------------------------------------------------------
    def _handle_pet_care(self, user_input: str) -> str:
        """Handle pet care questions using RAG system"""
        if self.rag_system is None:
            return "🐾 I'd love to help with pet care questions! However, the RAG system is not available right now. Please try again later."
        
        try:
            # Use RAG system to get detailed answer
            result = self.rag_system.ask(user_input)
            answer = result.get('answer', 'Sorry, I couldn\'t find information about that.')
            confidence = result.get('confidence', 0)
            
            # Debug: Log the result
            print(f"RAG Debug - Query: {user_input}")
            print(f"RAG Debug - Answer: {answer}")
            print(f"RAG Debug - Confidence: {confidence}")
            print(f"RAG Debug - Full result: {result}")
            
            # If answer is empty or generic, try to provide a better response
            if not answer or answer == "Sorry, I couldn't find information about that.":
                return self._provide_fallback_pet_care_answer(user_input)
            
            # Add confidence indicator if low
            if confidence < 0.7:
                return f"🐾 {answer}\n\n*Note: This answer has lower confidence. Please consult your veterinarian for specific medical advice.*"
            else:
                return f"🐾 {answer}"
                
        except Exception as e:
            # Fallback to simple responses if RAG fails
            lower_input = user_input.lower()
            
            if any(word in lower_input for word in ['feed', 'food', 'eating', 'diet']):
                return "🐾 For feeding your pet, I recommend:\n• High-quality commercial pet food appropriate for their age\n• Fresh water always available\n• Avoid human foods like chocolate, onions, and grapes\n• Consult your vet for specific dietary needs"
            
            elif any(word in lower_input for word in ['vaccine', 'vaccination', 'shots']):
                return "🐾 Vaccination schedule:\n• Puppies: 6-8 weeks, then every 3-4 weeks until 16 weeks\n• Kittens: 6-8 weeks, then every 3-4 weeks until 16 weeks\n• Adult pets: Annual boosters\n• Always consult your veterinarian for the best schedule"
            
            elif any(word in lower_input for word in ['groom', 'bath', 'clean']):
                return "🐾 Grooming tips:\n• Brush regularly to prevent matting\n• Bathe monthly or as needed\n• Trim nails carefully\n• Clean ears and teeth regularly\n• Use pet-safe products only"
            
            elif any(word in lower_input for word in ['exercise', 'walk', 'play']):
                return "🐾 Exercise recommendations:\n• Dogs: Daily walks and playtime\n• Cats: Interactive toys and climbing structures\n• Adjust activity level to your pet's age and health\n• Always supervise outdoor activities"
            
            else:
                return self._provide_fallback_pet_care_answer(user_input)
    
    def _provide_fallback_pet_care_answer(self, user_input: str) -> str:
        """Provide fallback pet care answers when RAG system fails"""
        lower_input = user_input.lower()
        
        if any(word in lower_input for word in ['feed', 'food', 'eating', 'diet', 'nutrition']):
            return "🐾 **Feeding Your Pet:**\n\n• **Puppies/Kittens**: Feed 3-4 times daily with high-quality puppy/kitten food\n• **Adult pets**: Feed 2 times daily with age-appropriate food\n• **Fresh water** should always be available\n• **Avoid human foods** like chocolate, onions, grapes, and avocado\n• **Consult your vet** for specific dietary needs and portion sizes\n\n*For personalized feeding advice, please consult your veterinarian.*"
        
        elif any(word in lower_input for word in ['vaccine', 'vaccination', 'shots', 'immunization']):
            return "🐾 **Vaccination Schedule:**\n\n• **Puppies**: Start at 6-8 weeks, then every 3-4 weeks until 16 weeks old\n• **Kittens**: Start at 6-8 weeks, then every 3-4 weeks until 16 weeks old\n• **Adult pets**: Annual booster vaccinations\n• **Core vaccines**: Rabies, DHPP (dogs), FVRCP (cats)\n• **Always consult your veterinarian** for the best vaccination schedule\n\n*Vaccination requirements may vary by location and pet health.*"
        
        elif any(word in lower_input for word in ['groom', 'bath', 'clean', 'hygiene']):
            return "🐾 **Grooming Tips:**\n\n• **Brushing**: Regular brushing prevents matting and reduces shedding\n• **Bathing**: Monthly or as needed (use pet-safe shampoo)\n• **Nail trimming**: Trim nails carefully to avoid cutting the quick\n• **Ear cleaning**: Clean ears regularly with vet-approved solution\n• **Dental care**: Brush teeth daily with pet toothpaste\n• **Use only pet-safe products** for grooming\n\n*Regular grooming keeps your pet healthy and comfortable.*"
        
        elif any(word in lower_input for word in ['exercise', 'walk', 'play', 'activity']):
            return "🐾 **Exercise & Play:**\n\n• **Dogs**: Daily walks and active playtime (30-60 minutes)\n• **Cats**: Interactive toys, climbing structures, and play sessions\n• **Adjust activity** to your pet's age, breed, and health\n• **Mental stimulation**: Puzzle toys and training exercises\n• **Always supervise** outdoor activities\n• **Weather considerations**: Avoid extreme heat or cold\n\n*Regular exercise keeps pets physically and mentally healthy.*"
        
        elif any(word in lower_input for word in ['train', 'training', 'behavior', 'obedience']):
            return "🐾 **Training Tips:**\n\n• **Start early**: Begin training as soon as you bring your pet home\n• **Positive reinforcement**: Reward good behavior with treats and praise\n• **Consistency**: Use the same commands and rules consistently\n• **Patience**: Training takes time and repetition\n• **Socialization**: Expose pets to different people, animals, and environments\n• **Professional help**: Consider obedience classes for complex issues\n\n*Training strengthens the bond between you and your pet.*"
        
        elif any(word in lower_input for word in ['health', 'sick', 'illness', 'symptoms']):
            return "🐾 **Pet Health Signs:**\n\n• **Watch for changes**: Appetite, energy level, behavior, or bathroom habits\n• **Common signs**: Lethargy, loss of appetite, vomiting, diarrhea, coughing\n• **Emergency signs**: Difficulty breathing, severe pain, unconsciousness\n• **Regular checkups**: Annual vet visits for preventive care\n• **Immediate vet care**: For any concerning symptoms\n\n*Always consult your veterinarian for health concerns.*"
        
        else:
            return "🐾 I'd love to help with pet care questions! For specific medical concerns, feeding advice, grooming tips, or training help, please consult your veterinarian. They can provide personalized guidance based on your pet's individual needs."

    # -----------------------------------------------------------------------
    # FIND-PET HANDLER
    # -----------------------------------------------------------------------
    def _handle_find_pet(self, user_input: str) -> str:
        ents = self._extract_entities(user_input)

        if "NOTICE" in ents:
            # If the user explicitly mentions "pet", guide them into the adoption flow
            if "pet" in user_input.lower():
                return (
                    "Got it! You're looking to adopt a pet. "
                    "Could you tell me what kind of pet and which state you're in? "
                    "For example: 'a dog in Johor' or 'a cat in Penang'."
                )
            return ents["NOTICE"]

        return self._update_entities_and_respond(ents)
    
    # -----------------------------------------------------------------------
    # UNKNOWN / LOW-CONFIDENCE HANDLER
    # -----------------------------------------------------------------------
    def _handle_unknown(self, user_input: str) -> str:
        """Handles unclear or nonsense inputs gracefully."""

        if self.session.get("intent") == "find_pet":
                
            if len(user_input) > 2 and len(user_input.split()) <= 2:
                token = user_input.strip()
                pseudo = f"I want a {token} {self.session['entities'].get('PET_TYPE', 'pet')}"
                ents = self._extract_entities(pseudo)
                if not ents or "NOTICE" in ents:
                    ents = self._extract_entities(user_input)
                if ents and "NOTICE" not in ents:
                    return self._update_entities_and_respond(ents)

            # Second try, direct extraction
            ents = self._extract_entities(user_input)
            if "NOTICE" in ents:
                return ents["NOTICE"]
            if ents:
                return self._update_entities_and_respond(ents)

            # Graceful confusion fallback
            if any(self.session["entities"].get(e) for e in REQUIRED_ENTITIES):
                return (
                    "Hmm, I didn't quite catch that. "
                    "Could you tell me a bit more — like 'small cream dog' or 'female cat'?"
                )

            return (
                "I'm not sure I understood that. "
                "Could you tell me what kind of pet and which state you're in? "
                "For example: 'I'm looking for a cat in Johor'."
            )

        # Generic fallback
        return get_response("unknown")

    # -----------------------------------------------------------------------
    # ENTITY EXTRACTION & VALIDATION
    # -----------------------------------------------------------------------
    def _extract_entities(self, text: str):
        """Runs NER extraction, applies synonym-based fallbacks, and filters invalid or out-of-scope entities."""
        ents = self.ner_extractor.extract(text)

        # --- Quick keyword check for out-of-scope animals ---
        species_out_of_scope = [
            "hamster", "hamsters", "rabbit", "rabbits", "bird", "birds",
            "parrot", "parrots", "fish", "fishes", "snake", "turtle"
        ]
        if any(w in text.lower() for w in species_out_of_scope):
            return {
                "NOTICE": (
                    "Sorry, I currently only help with cats 🐱 and dogs 🐶. "
                    "Would you like to search for one of those instead?"
                )
            }

        # --- Basic sanity checks ---
        if not text.strip() or len(text.strip()) < 2:
            return {"NOTICE": get_response("unknown")}
        if len(text) > 4 and not any(ch in "aeiou" for ch in text.lower()):
            return {"NOTICE": get_response("unknown")}

        # --- Remove placeholders / irrelevant tokens ---
        if "PET_TYPE" in ents and ents["PET_TYPE"].lower() in ["one", "it", "animal", "pet"]:
            ents.pop("PET_TYPE")
        if "AGE" in ents and str(ents["AGE"]).lower() in ["one", "1", "single", "johor"]:
            ents.pop("AGE")

        # --- Restrict supported species ---
        if "PET_TYPE" in ents and ents["PET_TYPE"].lower() not in ["dog", "cat"]:
            return {
                "NOTICE": (
                    "Sorry, I currently only help with cats 🐱 and dogs 🐶. "
                    "Would you like to search for one of those instead?"
                )
            }

        # --- Breed validation ---
        valid_breeds = [b.lower() for b in SYNONYMS.keys()]
        if "BREED" in ents:
            breed_val = ents["BREED"].lower()
            species_terms = [
                "hamster", "rabbit", "bird", "parrot", "fish", "turtle", "snake", "guinea",
                "hamsters", "rabbits", "birds", "parrots", "fishes"
            ]
            if (
                breed_val in species_terms
                or (breed_val not in valid_breeds and not re.search(r"[aeiou]", breed_val))
            ):
                ents.pop("BREED")

        # --- Color keyword fallback using synonym map ---
        color_variants = []
        for canon, variants in SYNONYMS.items():
            if canon.lower() in [
                "black", "white", "brown", "golden", "cream", "gray",
                "orange", "yellow", "blue", "red", "tabby", "calico", "tortoiseshell"
            ]:
                color_variants.extend([canon.lower()] + [v.lower() for v in variants])

        # Normalize text for color matching
        lower_text = autocorrect_text(text.lower())
        for color in color_variants:
            if re.search(rf"\b{re.escape(color)}\b", lower_text) or \
            re.search(rf"\b{re.escape(color)}\s*(color|colour)\b", lower_text) or \
            re.search(rf"\b(color|colour)\s*{re.escape(color)}\b", lower_text):
                ents["COLOR"] = canonicalize(color)
                break

        # --- drop total nonsense ---
        for k, v in list(ents.items()):
            if len(v) < 3 or not any(ch in "aeiou" for ch in v.lower()):
                ents.pop(k)

        # --- Prevent duplicate values across different entity types ---
        vals_seen = {}
        for k, v in list(ents.items()):
            v_low = v.lower().strip()
            if v_low in vals_seen.values():
                ents.pop(k)
            else:
                vals_seen[k] = v_low

        # --- No meaningful entities detected ---
        if not ents:
            return {"NOTICE": get_response("unknown")}

        # --- Remove redundant color if it's part of breed name ---
        if "BREED" in ents and "COLOR" in ents:
            if ents["COLOR"].lower() in ents["BREED"].lower():
                ents.pop("COLOR")

        return ents

    # -----------------------------------------------------------------------
    # UPDATE SESSION + RESPOND
    # -----------------------------------------------------------------------
    def _update_entities_and_respond(self, ents: dict) -> str:
        if "NOTICE" in ents:
            return ents["NOTICE"]

        confirm = []
        old_pet = self.session["entities"].get("PET_TYPE", "").lower()
        new_pet = ents.get("PET_TYPE", "").lower()

        if new_pet:
            if old_pet and new_pet != old_pet:
                for k in ["BREED", "FURLENGTH"]:
                    self.session["entities"].pop(k, None)
                confirm.append(f"Okay, updated pet type to {new_pet}.")
            self.session["entities"]["PET_TYPE"] = new_pet

        for k, v in ents.items():
            if k in ["PET_TYPE", "NOTICE"]:
                continue

            v = canonicalize(v)
            v = autocorrect_text(v)

            prev = self.session["entities"].get(k)
            readable = k.replace("_", " ").lower()
            if prev and prev != v:
                confirm.append(f"Okay, updated {readable} to {v}.")
            elif not prev:
                confirm.append(f"Added {readable}: {v}.")
            self.session["entities"][k] = v

        required_entities = REQUIRED_ENTITIES.copy()
        if "BREED" in self.session["entities"] or "BREED" in ents:
            if "PET_TYPE" in required_entities:
                required_entities.remove("PET_TYPE")

        missing = [e for e in required_entities if e not in self.session["entities"]]

        msg = " ".join(confirm)
        return f"{msg} {self.ask_for(missing[0])}" if missing else f"{msg} {self._confirm_and_search()}"

    # -----------------------------------------------------------------------
    # FINAL SEARCH MESSAGE
    # -----------------------------------------------------------------------
    def _confirm_and_search(self) -> str:
        ents = self.session["entities"]
        pet = ents.get("PET_TYPE", "pet")
        state = ents.get("STATE", "your area")
        details = [v for v in [
            ents.get("SIZE"), ents.get("COLOR"), ents.get("GENDER"),
            ents.get("AGE"), ents.get("FURLENGTH"), ents.get("BREED")
        ] if v]
        desc = " ".join(details + [pet])
        if not ents.get("BREED"):
            desc += "s"
        
        # Actually perform the pet search if we have the function
        if self.pet_search_func and self.azure_components:
            try:
                # Build search query from entities
                query_parts = []
                if ents.get("BREED"):
                    query_parts.append(ents["BREED"])
                if ents.get("PET_TYPE"):
                    query_parts.append(ents["PET_TYPE"])
                if ents.get("COLOR"):
                    query_parts.append(ents["COLOR"])
                if ents.get("SIZE"):
                    query_parts.append(ents["SIZE"])
                if ents.get("GENDER"):
                    query_parts.append(ents["GENDER"])
                if ents.get("STATE"):
                    query_parts.append(ents["STATE"])
                
                search_query = " ".join(query_parts)
                
                # Perform the search
                results, error = self.pet_search_func(search_query, self.azure_components)
                
                if results is not None and len(results) > 0:
                    # Format the results with photos
                    result_text = f"Found {len(results)} pets matching your criteria:\n\n"
                    for i, (_, pet) in enumerate(results.head(5).iterrows(), 1):
                        result_text += f"**{i}. {pet.get('name', 'Unnamed')}**\n"
                        result_text += f"   - {pet.get('animal', 'Unknown')} ({pet.get('breed', 'Mixed breed')})\n"
                        result_text += f"   - {pet.get('gender', 'Unknown')}, {pet.get('age_months', 'Unknown')} months old\n"
                        result_text += f"   - {pet.get('state', 'Unknown location')}\n"
                        
                        # Add photo if available
                        photo_links = pet.get('photo_links', '')
                        if photo_links and photo_links != '':
                            try:
                                import ast
                                if isinstance(photo_links, str):
                                    photos = ast.literal_eval(photo_links)
                                else:
                                    photos = photo_links
                                
                                if photos and len(photos) > 0:
                                    # Take the first photo
                                    first_photo = photos[0]
                                    result_text += f"   - 📸 [View Photo]({first_photo})\n"
                            except:
                                # If parsing fails, skip photo
                                pass
                        
                        if pet.get('description_clean'):
                            desc_short = pet['description_clean'][:100] + "..." if len(pet['description_clean']) > 100 else pet['description_clean']
                            result_text += f"   - {desc_short}\n"
                        result_text += "\n"
                    
                    if len(results) > 5:
                        result_text += f"... and {len(results) - 5} more pets available!\n"
                    
                    return result_text
                else:
                    return f"Sorry, I couldn't find any pets matching your criteria. {error or 'Try adjusting your search terms.'}"
                    
            except Exception as e:
                return f"Got it! Searching for {desc} in {state}... (Search temporarily unavailable: {str(e)})"
        else:
            return f"Got it! Searching for {desc} in {state}..."

    # -----------------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------------
    def _is_new_intent(self, intent, prev_intent):
        return (
            intent not in ["unknown", None]
            and prev_intent not in ["unknown", None]
            and intent != prev_intent
        )

    def ask_for(self, entity: str) -> str:
        prompts = {
            "PET_TYPE": "Are you looking for a dog or a cat?",
            "STATE": "Which state or area are you in?",
            "BREED": "Do you have a preferred breed?",
            "COLOR": "Any color preference?",
            "SIZE": "Do you prefer small or large pets?",
            "GENDER": "Male or female?",
            "AGE": "Puppy/kitten or adult?",
            "FURLENGTH": "Do you prefer short or long fur?",
        }
        return prompts.get(entity, f"Could you tell me the {entity.lower()}?")

    def _get_greeting(self) -> str:
        return (
            "Hello! 👋 I can help you find cats 🐱 or dogs 🐶 for adoption, "
            "or answer pet care questions.\n"
            "You can say things like:\n"
            "• I'm looking for a dog in Johor\n"
            "• How to care for a cat?\n"
            "So, what would you like to do today?"
        )

    def reset(self) -> str:
        self.session = {"intent": None, "entities": {}, "greeted": False}
        return self._get_greeting()


# ---------------------------------------------------------------------------
# LOCAL TESTING
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bot = ChatbotPipeline()
    print("Bot:", bot.handle_message(""))
    while True:
        msg = input("You: ")
        if msg.lower() in ["exit", "quit", "end"]:
            print("Bot: Goodbye! 👋")
            break
        if msg.lower() in ["restart", "reset", "new chat"]:
            print("Bot:", bot.reset())
            continue
        print("Bot:", bot.handle_message(msg))
