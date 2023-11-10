def query_discovery(input):
    from dotenv import load_dotenv
    import os
    from ibm_watson import DiscoveryV2
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    load_dotenv()
    discovery_api_key = os.getenv("WD_API_KEY", None)
    discovery_url = os.getenv("WD_SERVICE_URL", None)
    authenticator = IAMAuthenticator(discovery_api_key)
    discovery = DiscoveryV2(
    version='2020-08-30',
    authenticator=authenticator
    )

    discovery.set_service_url(discovery_url)

    response = discovery.query(
        project_id='0f1085c1-29eb-4fd3-b2fd-6553944cfce3',
        natural_language_query= input,
        count=2
    ).get_result()
    
    
    concatenated_text=""
    for passage in response["results"][0]["document_passages"]:
        concatenated_text+=passage["passage_text"]+'\n\n'
    
    return concatenated_text





def watsonx_call(question,discovery):
    model_id = "meta-llama/llama-2-70b-chat"
    parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 20,
    "repetition_penalty": 1
}
    import os
    from dotenv import load_dotenv
    load_dotenv()
    project_id = os.getenv("WX_PROJECT_ID", None)
    from ibm_watson_machine_learning.foundation_models import Model

    model = Model(
	model_id = model_id,
	params = parameters,
	credentials = {
		"url" : "https://us-south.ml.cloud.ibm.com",
		"apikey" : os.getenv("IAM_API_KEY", None)
	},
	project_id = project_id,
	
	)
    prompt_input = f"""Extract only the {question} from the following text and output only the value

Input:{discovery}

Output:
"""
    print("Submitting generation request...")
    generated_response = model.generate_text(prompt=prompt_input)
   
    return generated_response






import streamlit as st
st.header('Form 16 Entity Extractor')
question=st.text_input('Enter the entity')
button=st.button('Generate')

if button:
    discovery=query_discovery(question)
    st.subheader('Discovery response')
    print(discovery)
    st.write(discovery)
    output=watsonx_call(question,discovery)
    print(output)
    st.subheader('watsonx response')
    st.write(output)
