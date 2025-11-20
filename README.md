# Airbnb Price Prediction in London

## Contributors

- Adrien LORY
- Viviam BAIZ
- Lucas STEFANI

---

## 1. Business Challenge

The objective of this project is to **predict the nightly price of Airbnb listings in London**.

This matters because:

- New or occasional hosts often have **no reference** for how much they should charge.
- Overpricing leads to **low occupancy**, while underpricing leads to **lost revenue**.
- A data-driven pricing model can help hosts set **fair and competitive prices** based on:
  - Location
  - Capacity (bedrooms, bathrooms, number of guests)
  - Host experience
  - Reviews and ratings
  - Amenities and availability

Our goal is to build a supervised learning model that takes listing characteristics as inputs and outputs a **recommended price per night**.

---

## 2. Dataset Description



### Main feature categories

- **Host features**
  - `host_days_active` (days since host joined)
  - `host_is_superhost`
  - `host_acceptance_rate`
- **Listing structure**
  - `accommodates`
  - `bedrooms`, `bathrooms`
  - `minimum_nights`, `maximum_nights`
- **Availability**
  - `availability_30`, `availability_365`
- **Demand / reputation**
  - `number_of_reviews`
  - `reviews_per_month`
  - `review_scores_rating`
- **Amenities**
  - `num_amenities`
  - Binary flags for key amenities (wifi, kitchen, washer, etc.)
- **Location**
  - Neighbourhood prestige groups: `neigh_Budget`, `neigh_Mid_Range`, `neigh_High_End`, `neigh_Prime`

The version used for modeling is already cleaned and exported as:

```text
data/airbnb_london_cleaned.csv
