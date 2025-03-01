"""
title: CustomerRetrievalPipeline
author: Insurance Claims Team
date: 2025-03-01
version: 1.0
license: MIT
description: Retrieves customer information based on claim form data for verification
requirements: pydantic
"""

from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import os

class Pipeline:
    class Valves(BaseModel):
        CUSTOMER_DATA_PATH: str = os.path.join(os.path.dirname(__file__), "demo_customer.json")
        MATCH_THRESHOLD: float = 0.7  # Confidence threshold for matches
        HIGHLIGHT_DISCREPANCIES: bool = True
    
    def __init__(self):
        self.name = "Customer Retrieval Pipeline"
        self.valves = self.Valves()
        self.customer_data = None
    
    async def on_startup(self):
        """Load the demo customer data when the pipeline starts"""
        try:
            if os.path.exists(self.valves.CUSTOMER_DATA_PATH):
                with open(self.valves.CUSTOMER_DATA_PATH, 'r') as f:
                    self.customer_data = json.load(f)
            else:
                # Create a demo customer if file doesn't exist
                self.customer_data = self._create_demo_customer()
                
                # Save the demo customer for future use
                os.makedirs(os.path.dirname(self.valves.CUSTOMER_DATA_PATH), exist_ok=True)
                with open(self.valves.CUSTOMER_DATA_PATH, 'w') as f:
                    json.dump(self.customer_data, f, indent=2)
            
            print(f"Customer Retrieval Pipeline started with demo customer: {self.customer_data['customer_information']['full_name']}")
        except Exception as e:
            print(f"Error loading customer data: {str(e)}")
    
    async def on_shutdown(self):
        """Clean up resources when the pipeline shuts down"""
        self.customer_data = None
        print("Customer Retrieval Pipeline shut down.")
    
    def pipe(self, claim_data, model_id=None, messages=None, body=None):
        """
        Match claim form data against customer records and return customer profile.
        
        Args:
            claim_data: Dictionary containing extracted claim form information
            
        Returns:
            Dictionary with customer data and match information
        """
        try:
            if not self.customer_data:
                return {
                    "status": "error",
                    "message": "Customer database not initialized"
                }
            
            # Extract search parameters from claim data
            search_params = self._extract_search_params(claim_data)
            
            # Calculate match score
            match_results = self._calculate_match(search_params)
            
            # Check if match exceeds threshold
            if match_results["match_score"] < self.valves.MATCH_THRESHOLD:
                return {
                    "status": "no_match",
                    "message": "No customer record matched the claim information",
                    "search_params": search_params,
                    "match_score": match_results["match_score"]
                }
            
            # Prepare the response
            customer_profile = self.customer_data.copy()
            
            # Add match information
            result = {
                "status": "match_found",
                "customer_profile": customer_profile,
                "match_score": match_results["match_score"],
                "matched_fields": match_results["matched_fields"],
                "discrepancies": match_results["discrepancies"],
                "verification_summary": self._create_verification_summary(match_results)
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error retrieving customer data: {str(e)}"
            }
    
    def _extract_search_params(self, claim_data):
        """Extract search parameters from claim data"""
        search_params = {}
        
        # Extract policyholder information
        if "policyholder" in claim_data:
            policyholder = claim_data["policyholder"]
            if "name" in policyholder:
                search_params["full_name"] = policyholder["name"]
            if "policy_number" in policyholder:
                search_params["policy_number"] = policyholder["policy_number"]
            if "phone" in policyholder:
                search_params["phone_number"] = policyholder["phone"]
            if "email" in policyholder:
                search_params["email_address"] = policyholder["email"]
        
        # Extract vehicle information
        if "vehicle" in claim_data:
            vehicle = claim_data["vehicle"]
            if "make" in vehicle:
                search_params["vehicle_make"] = vehicle["make"]
            if "model" in vehicle:
                search_params["vehicle_model"] = vehicle["model"]
            if "year" in vehicle:
                search_params["vehicle_year"] = vehicle["year"]
            if "vin" in vehicle:
                search_params["vehicle_vin"] = vehicle["vin"]
            if "license_plate" in vehicle:
                search_params["license_plate"] = vehicle["license_plate"]
        
        return search_params
    
    def _calculate_match(self, search_params):
        """
        Calculate match score and identify discrepancies.
        
        For demo purposes, this will do a simple field comparison with the demo customer.
        """
        matched_fields = []
        discrepancies = []
        total_weight = 0
        matched_weight = 0
        
        # Field weights for scoring (importance of each field)
        weights = {
            "policy_number": 10,
            "full_name": 8,
            "email_address": 7,
            "phone_number": 6,
            "vehicle_vin": 10,
            "license_plate": 9,
            "vehicle_make": 3,
            "vehicle_model": 3,
            "vehicle_year": 2
        }
        
        # Extract customer info for comparison
        customer_info = {
            "full_name": self.customer_data["customer_information"]["full_name"],
            "email_address": self.customer_data["customer_information"]["contact_information"]["email"],
            "phone_number": self.customer_data["customer_information"]["contact_information"]["phone"]
        }
        
        # Add policy information
        if self.customer_data["policy_summary"]["policies"]:
            customer_info["policy_number"] = self.customer_data["policy_summary"]["policies"][0]["policy_id"]
        
        # Add vehicle information if available
        if self.customer_data["vehicle_summary"]["vehicles"]:
            vehicle = self.customer_data["vehicle_summary"]["vehicles"][0]
            make_model = vehicle["make_model"].split()
            customer_info["vehicle_make"] = make_model[0] if make_model else ""
            customer_info["vehicle_model"] = make_model[1] if len(make_model) > 1 else ""
            customer_info["vehicle_year"] = vehicle["make_model"].split("(")[-1].strip(")") if "(" in vehicle["make_model"] else ""
            customer_info["vehicle_vin"] = vehicle["vin"]
            customer_info["license_plate"] = vehicle["license_plate"]
        
        # Compare fields and calculate score
        for field, weight in weights.items():
            if field in search_params and field in customer_info:
                total_weight += weight
                
                # Case-insensitive comparison for text fields
                claim_value = str(search_params[field]).lower().strip()
                db_value = str(customer_info[field]).lower().strip()
                
                if claim_value == db_value:
                    matched_fields.append(field)
                    matched_weight += weight
                else:
                    discrepancies.append({
                        "field": field,
                        "claim_value": search_params[field],
                        "database_value": customer_info[field]
                    })
        
        # Calculate final score (0-1 range)
        match_score = matched_weight / total_weight if total_weight > 0 else 0
        
        return {
            "match_score": match_score,
            "matched_fields": matched_fields,
            "discrepancies": discrepancies
        }
    
    def _create_verification_summary(self, match_results):
        """Create a human-readable verification summary"""
        score = match_results["match_score"]
        matched = len(match_results["matched_fields"])
        discrepancies = len(match_results["discrepancies"])
        
        if score >= 0.9:
            confidence = "High confidence match"
        elif score >= 0.8:
            confidence = "Good confidence match"
        elif score >= 0.7:
            confidence = "Moderate confidence match"
        else:
            confidence = "Low confidence match"
        
        summary = f"{confidence} ({score:.2f}). {matched} fields matched with {discrepancies} discrepancies."
        
        if discrepancies > 0:
            summary += " Please review the discrepancies carefully."
            
            # Add specific discrepancy details
            summary += " Discrepancies found in: "
            discrepancy_fields = [d["field"] for d in match_results["discrepancies"]]
            summary += ", ".join(discrepancy_fields)
        
        return summary
    
    def _create_demo_customer(self):
        """Create a demo customer for testing"""
        return {
            "customer_information": {
                "customer_id": "C123456",
                "full_name": "Michael Johnson",
                "age": 42,
                "contact_information": {
                    "email": "michael.johnson@example.com",
                    "phone": "555-867-5309",
                    "address": {
                        "street": "123 Pine Dr",
                        "city": "Springfield",
                        "state": "IL",
                        "zip_code": "62704"
                    }
                },
                "driver_information": {
                    "license_number": "DL987654D",
                    "license_status": "Valid",
                    "license_expiry": "2027-05-15",
                    "days_to_expiry": 795,
                    "years_driving": 24
                },
                "personal_information": {
                    "date_of_birth": "1982-08-24",
                    "gender": "Male",
                    "marital_status": "Married",
                    "occupation": "Software Engineer"
                }
            },
            "policy_summary": {
                "total_policies": 1,
                "active_policies": 1,
                "total_premium": "$1,450.00",
                "policies": [
                    {
                        "policy_id": "POL-573829",
                        "policy_type": "Comprehensive",
                        "status": "Active",
                        "coverage_period": "2024-08-15 to 2025-08-15",
                        "days_to_expiry": 165,
                        "premium": "$1,450.00",
                        "payment_frequency": "Monthly",
                        "payment_status": "Current",
                        "coverage_limits": {
                            "bodily_injury": "250000/500000",
                            "property_damage": "100000",
                            "collision": "$500 deductible",
                            "comprehensive": "$500 deductible"
                        },
                        "discounts": ["Multi-policy", "Good driver", "Anti-theft device"]
                    }
                ]
            },
            "vehicle_summary": {
                "total_vehicles": 1,
                "vehicles": [
                    {
                        "vehicle_id": "V123456",
                        "make_model": "Honda Accord (2022)",
                        "vin": "1HGCM82633A123456",
                        "type": "Sedan",
                        "license_plate": "ABC-1234",
                        "mileage": 25000,
                        "fuel_type": "Gasoline",
                        "safety_features": ["Airbags", "Anti-lock Brakes", "Backup Camera", "Lane Departure Warning"]
                    }
                ]
            },
            "claims_summary": {
                "total_claims": 1,
                "recent_claims": 1,
                "claims": [
                    {
                        "claim_id": "CL-987654",
                        "date": "2024-09-18",
                        "status": "Paid",
                        "amount": "$2,200.00",
                        "description": "Front bumper damage from collision in parking lot",
                        "fault": "Not-At-Fault",
                        "police_report": "N/A",
                        "is_recent": True
                    }
                ]
            },
            "risk_assessment": {
                "risk_score": 85,
                "risk_category": "Low Risk",
                "fraud_risk_indicator": 0.12
            }
        }