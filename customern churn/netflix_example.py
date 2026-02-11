"""
Netflix Churn Prediction Example
================================
Shows how to adapt the churn prediction model for Netflix customers
Using similar structure to Telco but with Netflix-specific features
"""

import pandas as pd
from churn_predict import predict_churn

# ==================== NETFLIX CUSTOMER EXAMPLES ====================

print("="*70)
print("NETFLIX CHURN PREDICTION - CUSTOMER SCENARIOS")
print("="*70)

# ✅ Customer 1: Long-term, Active Watcher (LOW RISK)
print("\n📊 CUSTOMER 1: Long-term, Active Subscriber")
print("-" * 70)

netflix_customer_1 = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 36,  # 3 years subscription
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",  # Premium internet = premium viewer
    "OnlineSecurity": "Yes",  # Security conscious = committed user
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",  # Using support = engaged
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",  # Long-term commitment
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Automatic",  # Automated = less friction
    "MonthlyCharges": 119.85,  # Premium tier
    "TotalCharges": 4314.6  # High lifetime value
}

result_1 = predict_churn(netflix_customer_1)
if result_1:
    print(f"Name: Long-term Premium Subscriber")
    print(f"Status: {result_1['prediction']}")
    print(f"Churn Risk: {result_1['churn_probability']:.1%}")
    print(f"Analysis: 3-year customer, premium services, automated payment")
    print(f"✓ RETENTION STRATEGY: Maintain service quality, exclusive content")

# ❌ Customer 2: New, Budget Subscriber (HIGH RISK)
print("\n" + "="*70)
print("\n📊 CUSTOMER 2: New Budget Subscriber")
print("-" * 70)

netflix_customer_2 = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,  # 2 months - very new
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",  # Basic internet
    "OnlineSecurity": "No",  # Not investing in add-ons = less committed
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",  # Not seeking support
    "StreamingTV": "No",  # Only streaming movies
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",  # No commitment
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",  # Manual payment = higher friction
    "MonthlyCharges": 9.99,  # Basic tier
    "TotalCharges": 19.98  # Low lifetime value
}

result_2 = predict_churn(netflix_customer_2)
if result_2:
    print(f"Name: New Budget Subscriber")
    print(f"Status: {result_2['prediction']}")
    print(f"Churn Risk: {result_2['churn_probability']:.1%}")
    print(f"Analysis: Just joined, basic tier, month-to-month, no add-ons")
    print(f"⚠ RETENTION STRATEGY: Send personalized content recs, offer free trial perks")

# ⚠️ Customer 3: At-Risk Subscriber (MEDIUM-HIGH RISK)
print("\n" + "="*70)
print("\n📊 CUSTOMER 3: At-Risk Subscriber")
print("-" * 70)

netflix_customer_3 = {
    "gender": "Male",
    "SeniorCitizen": 1,  # Senior - price sensitive
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 6,  # 6 months - early stage
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",  # Not adding services
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",  # No engagement
    "StreamingTV": "No",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",  # Still uncommitted
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 15.99,  # Standard tier
    "TotalCharges": 95.94  # Low-medium lifetime value
}

result_3 = predict_churn(netflix_customer_3)
if result_3:
    print(f"Name: At-Risk Senior Subscriber")
    print(f"Status: {result_3['prediction']}")
    print(f"Churn Risk: {result_3['churn_probability']:.1%}")
    print(f"Analysis: Senior, 6 months, no premium services, month-to-month")
    print(f"🎯 RETENTION STRATEGY: Senior discount, free premium month trial")

# ✅ Customer 4: Growing Power User (LOW-MEDIUM RISK)
print("\n" + "="*70)
print("\n📊 CUSTOMER 4: Growing Power User")
print("-" * 70)

netflix_customer_4 = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 18,  # 18 months
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",  # Premium
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "One year",  # Growing commitment
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Automatic",
    "MonthlyCharges": 89.99,  # Premium tier
    "TotalCharges": 1619.82
}

result_4 = predict_churn(netflix_customer_4)
if result_4:
    print(f"Name: Growing Power User")
    print(f"Status: {result_4['prediction']}")
    print(f"Churn Risk: {result_4['churn_probability']:.1%}")
    print(f"Analysis: Family account, 18 months, premium services, 1-year contract")
    print(f"✓ RETENTION STRATEGY: Family features, exclusive content, early access")

# ==================== BUSINESS INSIGHTS ====================

print("\n" + "="*70)
print("📈 CHURN PREDICTION INSIGHTS FOR NETFLIX")
print("="*70)

insights = """
KEY FACTORS PREDICTING CHURN:
┌─────────────────────────────────────────────────────────────┐
│ HIGH RISK INDICATORS:                                       │
│ • New customers (tenure < 3 months)                         │
│ • Month-to-month contract (no commitment)                   │
│ • Basic/budget tier only (no premium services)              │
│ • No add-on services (OnlineSecurity, DeviceProtection)     │
│ • Manual payment methods (higher friction)                  │
│ • Low monthly charges (price-sensitive customers)           │
│                                                             │
│ LOW RISK INDICATORS:                                        │
│ • Long tenure (>12 months)                                  │
│ • Long-term contract (one year or two year)                 │
│ • Premium tier with add-ons                                 │
│ • Automatic payment setup                                   │
│ • High monthly charges (committed premium users)            │
│ • Multiple services used (StreamingTV + Movies + Security)  │
│                                                             │
│ RETENTION STRATEGIES BY SEGMENT:                            │
│ • NEW USERS (0-3 months): Free trial features, onboarding  │
│ • BUDGET USERS: Discounts, upgrade incentives              │
│ • AT-RISK: Win-back campaigns, exclusive content            │
│ • POWER USERS: VIP treatment, early feature access          │
└─────────────────────────────────────────────────────────────┘
"""

print(insights)

# ==================== NETFLIX-SPECIFIC ADAPTATIONS ====================

print("\n" + "="*70)
print("🎬 NETFLIX-SPECIFIC MODEL ADAPTATIONS")
print("="*70)

adaptations = """
MAPPING TELCO FEATURES TO NETFLIX:

TELCO FEATURE          →  NETFLIX EQUIVALENT
─────────────────────────────────────────────────────
tenure                 →  Subscription duration
PhoneService           →  Multi-device support
InternetService        →  Streaming quality tier
OnlineSecurity         →  Parental controls/Premium features
OnlineBackup           →  Download for offline feature
DeviceProtection       →  Multi-screen capability
TechSupport            →  Customer support engagement
StreamingTV            →  Standard library access
StreamingMovies        →  Standard library access
Contract               →  Subscription plan type
MonthlyCharges         →  Subscription price tier
TotalCharges           →  Lifetime customer value

NETFLIX-NATIVE FEATURES YOU COULD ADD:
• content_watched_ratio: % of recommendations watched
• genres_watched: Diversity of content consumption
• days_since_last_watch: Recency of engagement
• avg_watch_time: Session duration
• profile_count: Multiple household members
• download_feature_used: Offline viewing usage
"""

print(adaptations)

# ==================== USAGE FOR NETFLIX TEAM ====================

print("\n" + "="*70)
print("💼 HOW NETFLIX WOULD USE THIS MODEL")
print("="*70)

use_cases = """
1️⃣ REAL-TIME CHURN SCORING
   • When customer logs in, predict churn risk
   • Show targeted offers if high-risk detected
   • Recommend content to increase engagement

2️⃣ BATCH PROCESSING
   • Run weekly on all active customers
   • Identify at-risk segments
   • Trigger automated retention campaigns

3️⃣ A/B TESTING
   • Test different retention strategies by risk segment
   • Measure impact on churn reduction
   • Refine model with outcomes

4️⃣ BUSINESS INTELLIGENCE
   • Track churn trends by region/demographics
   • Identify pricing elasticity
   • Optimize content investment by segment

5️⃣ CUSTOMER SUPPORT
   • Flag high-value at-risk customers
   • Empower support team with intervention options
   • Track intervention success rates
"""

print(use_cases)

# ==================== COMPARISON TABLE ====================

print("\n" + "="*70)
print("📊 CUSTOMER COMPARISON TABLE")
print("="*70)

comparison_data = {
    'Customer': ['Premium Long-term', 'Budget New', 'At-Risk Senior', 'Growing Power'],
    'Tenure (months)': [36, 2, 6, 18],
    'Monthly Cost': ['$119.85', '$9.99', '$15.99', '$89.99'],
    'Contract': ['2-year', 'Month-to-month', 'Month-to-month', '1-year'],
    'Premium Services': ['All (4/4)', 'None (0/4)', 'None (0/4)', 'All (4/4)'],
    'Predicted Status': [result_1['prediction'], result_2['prediction'], result_3['prediction'], result_4['prediction']],
    'Churn Risk': [f"{result_1['churn_probability']:.1%}", f"{result_2['churn_probability']:.1%}", 
                   f"{result_3['churn_probability']:.1%}", f"{result_4['churn_probability']:.1%}"],
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

print("\n" + "="*70)
print("✅ Model successfully integrated for Netflix use case!")
print("="*70 + "\n")
