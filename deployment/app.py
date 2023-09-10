import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import json


with open('model_scaler.pkl', 'rb') as file_2:
  scaler = pickle.load(file_2)
  
with open('model_pca.pkl', 'rb') as file_3:
  pca_model = pickle.load(file_3)

with open('model_kmeans.pkl', 'rb') as file_4:
  model_kmeans = pickle.load(file_4)

st.title('Understanding Customers through Behavior-Based Clustering')
st.set_option('deprecation.showPyplotGlobalUse', False)

def run():
  
  hide_streamlit_style = """
              <style>
              #MainMenu {visibility: hidden;}
              footer {visibility: hidden;}
              </style>
              """
  st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
  
  with st.form(key='Prediction form'):
      age = st.number_input('Select Number of Age', 0, 100, step=1)
      account_id = st.number_input('Select Number of Account ID', 0, 10000, step=1)
      card_id = st.number_input('Select Number of Card ID', 0, 10000, step=1)
      debit_amount = st.number_input('Select Number of Debit Amount', 0, 10000000, step=1)
      credit_amount = st.number_input('Select Number of Credit Amount', 0, 10000000, step=1)
      cash_withdrawal_total = st.number_input('Select Number of Total Cash Withdrawal', 0, 10000000, step=1)
      remittance_to_another_bank_total = st.number_input('Select Number of Total Remittance to another bank', 0, 10000000, step=1)
      credit_in_cash_total = st.number_input('Select Number of Total Credit in Cash', 0, 10000000, step=1)
      collection_from_another_bank_total = st.number_input('Select Number of Total Collection from Another Bank', 0, 10000000, step=1)
      credit_card_withdrawal_total = st.number_input('Select Number of Total Credit Card Withdrawal', 0, 10000000, step=1)
      total_transaction_amount = st.number_input('Select Number of Total Amount of Transaction', 0, 10000000, step=1)
      last_balance = st.number_input('Select Number of Last Balance', 0, 10000000, step=1)
      max_balance = st.number_input('Select Number of Max Balance', 0, 10000000, step=1)
      average_balance = st.number_input('Select Number of Average Balance', 0, 10000000, step=1)
      
      submitted = st.form_submit_button('Predict')
  
  data_inf = {
        'birth_number' : 999999,
        'district_id' : 1,
        'age' : age,
        'disp_id' : 1,
        'account_id' : account_id,
        'card_id' : card_id,
        'issued' : 931107,
        'debit_amount' : debit_amount,
        'credit_amount' : credit_amount,
        'CASH WITHDRAWAL_total' : cash_withdrawal_total,
        'REMITTANCE TO ANOTHER BANK_total' : remittance_to_another_bank_total,
        'CREDIT IN CASH_total' : credit_in_cash_total,
        'COLLECTION FROM ANOTHER BANK_total' : collection_from_another_bank_total,
        'CREDIT CARD WITHDRAWAL_total' : credit_card_withdrawal_total,
        'total_transaction_amount' : total_transaction_amount,
        'last_balance' : last_balance,
        'max_balance' : max_balance,
        'min_balance' : 0,
        'average_balance' : average_balance,
    }
  
  data_inf = pd.DataFrame([data_inf])
  num_cols = ['CASH WITHDRAWAL_total', 'CREDIT IN CASH_total', 'COLLECTION FROM ANOTHER BANK_total', 'max_balance', 'total_transaction_amount', 'average_balance', 'total_transaction_amount', 'credit_amount', 'debit_amount']
  num_data = data_inf[num_cols]
    
  if submitted:
        X_scaled = scaler.transform(num_data)
        num_pca = pca_model.transform(X_scaled)[:, :3]
        prediction_cluster = pd.DataFrame(model_kmeans.predict(num_pca))
        st.write(f'For account ID **{account_id}**, we recommend you')
        
        if prediction_cluster.iloc[0, 0] == 0:
            st.write('''
                     **Cluster 0 - High Activity Premium Package:**

                      This package is designed for customers who engage in high levels of banking activity and have substantial financial resources.

                      - Free Transaction Fee to Another Bank: Allow free transactions to and from accounts at other banks to accommodate frequent transactions.
                      - High Reward per Transaction: Offer rewards, cashback, or loyalty points for each transaction made.
                      - Comprehensive Checking and Savings Accounts: Provide feature-rich checking and savings accounts with high interest rates, suitable for high-volume transactions.
                      - Business Loans: Offer business loans with competitive terms and rates to support entrepreneurial ventures.
                      - Full Insurance Coverage: Include comprehensive business, personal, and life insurance options to protect assets and manage risk.
                      - Premium Investment Services: Extend professional investment advice, portfolio management, and access to exclusive investment opportunities.
                      - Personalized Financial Planning: Provide dedicated financial advisors for comprehensive financial planning and wealth management.
                      - Exclusive Banking Perks: Grant access to exclusive banking lounges, concierge services, and premium debit and credit cards.
                                          ''')
        elif prediction_cluster.iloc[0, 0] == 1:
            st.write('''
                     **Cluster 1 - Balanced Banking Package:**

                      This package is designed for customers with moderate financial activity and balanced banking needs.

                      - Reasonable Transaction Fees: Charge reasonable transaction fees while offering a certain number of free transactions each month.
                      - Moderate Reward Program: Provide a moderate rewards program for customers to benefit from their banking activity.
                      - Savings and Investment Accounts: Offer both high-yield savings accounts and access to a range of investment options.
                      - Retirement Planning: Include retirement planning services to help customers save for their future.
                      - Basic Insurance Coverage: Provide essential insurance coverage for peace of mind.
                      - Professional Investment Advice: Offer investment advice to help customers grow their wealth.
                      - Mobile Banking Convenience: Ensure convenient mobile banking access for easy account management.
                                       ''')
        elif prediction_cluster.iloc[0, 0] == 2:
            st.write('''
                     **Cluster 2 - High Balance Premium Package:**

                      This package is designed for customers who maintain high balances and engage in various banking activities.

                      - No Transaction Fees: Eliminate transaction fees entirely, including those for transfers to other banks.
                      - High-Interest Savings Account: Provide a high-interest savings account with premium interest rates.
                      - Access to Exclusive Investments: Offer access to exclusive investment opportunities with high potential returns.
                      - Comprehensive Insurance: Include comprehensive insurance coverage, including property, health, and life insurance.
                      - Tailored Investment Strategies: Provide personalized investment strategies based on individual goals and risk tolerance.
                      - Concierge Services: Offer premium concierge services for travel, dining, and lifestyle needs.
                      - Full Financial Planning: Provide comprehensive financial planning, estate planning, and tax optimization services.
                      ''')
        else:
            st.write('''
                     **Cluster 3 - Basic Banking Package:**

                        This package is designed to customers with minimal banking activity and straightforward needs.

                        - Low Transaction Fees: Charge minimal transaction fees for basic banking services.
                        - Basic Savings Account: Offer a basic savings account with competitive interest rates.
                        - Retirement Savings Guidance: Provide guidance on retirement savings options.
                        - Essential Insurance: Include essential insurance coverage for individuals and their families.
                        - Simplified Investment Solutions: Offer easy-to-understand, low-risk investment options.
                        - Budgeting Assistance: Provide budgeting tools and resources to help customers manage their finances effectively.
                        - Customer Education: Offer financial literacy resources to help customers make informed decisions.
                                          ''')
            
if __name__ == '__main__':
    run()