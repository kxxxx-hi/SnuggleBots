"""
Pet data management for SnuggleBots
Contains sample pet data with Azure Blob Storage image URLs
"""

from typing import List, Dict, Any
import random

# Sample pet data based on the provided image
SAMPLE_PETS = [
    {
        "id": 1,
        "name": "Buddy",
        "type": "Dog",
        "breed": "Golden Retriever",
        "age": "2 years",
        "gender": "Male",
        "size": "Medium",
        "description": "Friendly and playful, loves walks.",
        "image_url": "https://snugglebotsstorage.blob.core.windows.net/pets/buddy.jpg"
    },
    {
        "id": 2,
        "name": "Whiskers",
        "type": "Cat",
        "breed": "Siamese",
        "age": "1 year",
        "gender": "Female",
        "size": "Small",
        "description": "Quiet and affectionate, enjoys naps.",
        "image_url": "https://snugglebotsstorage.blob.core.windows.net/pets/whiskers.jpg"
    },
    {
        "id": 3,
        "name": "Max",
        "type": "Dog",
        "breed": "Labrador",
        "age": "3 years",
        "gender": "Male",
        "size": "Large",
        "description": "Energetic and loyal, good with kids.",
        "image_url": "https://snugglebotsstorage.blob.core.windows.net/pets/max.jpg"
    },
    {
        "id": 4,
        "name": "Luna",
        "type": "Cat",
        "breed": "Persian",
        "age": "2 years",
        "gender": "Female",
        "size": "Medium",
        "description": "Calm and gentle, perfect for families.",
        "image_url": "https://snugglebotsstorage.blob.core.windows.net/pets/luna.jpg"
    },
    {
        "id": 5,
        "name": "Rocky",
        "type": "Dog",
        "breed": "German Shepherd",
        "age": "4 years",
        "gender": "Male",
        "size": "Large",
        "description": "Protective and intelligent, great for experienced owners.",
        "image_url": "https://snugglebotsstorage.blob.core.windows.net/pets/rocky.jpg"
    },
    {
        "id": 6,
        "name": "Mittens",
        "type": "Cat",
        "breed": "Maine Coon",
        "age": "1 year",
        "gender": "Female",
        "size": "Large",
        "description": "Playful and social, loves attention.",
        "image_url": "https://snugglebotsstorage.blob.core.windows.net/pets/mittens.jpg"
    }
]

class PetRecommendationSystem:
    """Handles pet recommendations based on user queries"""
    
    def __init__(self):
        self.pets = SAMPLE_PETS
    
    def search_pets(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search for pets based on query keywords
        Returns matching pets with their information
        """
        query_lower = query.lower()
        matching_pets = []
        
        for pet in self.pets:
            # Check if query matches any pet attributes
            if (query_lower in pet["name"].lower() or
                query_lower in pet["type"].lower() or
                query_lower in pet["breed"].lower() or
                query_lower in pet["description"].lower() or
                query_lower in pet["size"].lower() or
                query_lower in pet["gender"].lower()):
                matching_pets.append(pet)
        
        # If no specific matches, return random pets
        if not matching_pets:
            matching_pets = random.sample(self.pets, min(max_results, len(self.pets)))
        else:
            matching_pets = matching_pets[:max_results]
        
        return matching_pets
    
    def get_all_pets(self) -> List[Dict[str, Any]]:
        """Get all available pets"""
        return self.pets
    
    def get_pet_by_id(self, pet_id: int) -> Dict[str, Any]:
        """Get a specific pet by ID"""
        for pet in self.pets:
            if pet["id"] == pet_id:
                return pet
        return None
