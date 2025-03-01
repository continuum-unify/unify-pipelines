import asyncio
import json
from customer_database import Pipeline  # Import the Pipeline class

def test_customer_pipeline():
    """Function to test the Customer Retrieval Pipeline"""
    
    # Initialize the pipeline
    pipeline = Pipeline()
    
    # Run the pipeline startup to load demo data
    asyncio.run(pipeline.on_startup())

    # ✅ Test Case 1: Perfect Match
    claim_data_match = {
        "policyholder": {
            "name": "Michael Johnson",
            "policy_number": "POL-573829",
            "phone": "555-867-5309",
            "email": "michael.johnson@example.com"
        },
        "vehicle": {
            "make": "Honda",
            "model": "Accord",
            "year": "2022",
            "vin": "1HGCM82633A123456",
            "license_plate": "ABC-1234"
        }
    }

    result_match = pipeline.pipe(claim_data_match)
    print("\n✅ Test Case 1: Perfect Match")
    print(json.dumps(result_match, indent=2))

    # ⚠️ Test Case 2: Partial Match (some discrepancies)
    claim_data_partial = {
        "policyholder": {
            "name": "Mike Johnson",  # Slight name variation
            "policy_number": "POL-573829",
            "phone": "555-867-5308",  # Different phone number
            "email": "mike.johnson@example.com"  # Different email
        },
        "vehicle": {
            "make": "Honda",
            "model": "Civic",  # Different model
            "year": "2022",
            "vin": "1HGCM82633A123456",
            "license_plate": "ABC-1234"
        }
    }

    result_partial = pipeline.pipe(claim_data_partial)
    print("\n⚠️ Test Case 2: Partial Match")
    print(json.dumps(result_partial, indent=2))

    # ❌ Test Case 3: No Match (completely different data)
    claim_data_no_match = {
        "policyholder": {
            "name": "John Doe",
            "policy_number": "POL-999999",
            "phone": "555-000-0000",
            "email": "john.doe@example.com"
        },
        "vehicle": {
            "make": "Toyota",
            "model": "Camry",
            "year": "2020",
            "vin": "5YJSA1E26MF123456",
            "license_plate": "XYZ-9876"
        }
    }

    result_no_match = pipeline.pipe(claim_data_no_match)
    print("\n❌ Test Case 3: No Match")
    print(json.dumps(result_no_match, indent=2))

    # Run pipeline shutdown
    asyncio.run(pipeline.on_shutdown())

# Run the test
if __name__ == "__main__":
    test_customer_pipeline()
