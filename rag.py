import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import fitz
import io
from PIL import Image

import time

import os
from dotenv import load_dotenv

load_dotenv()

# Supress warnings
import warnings
warnings.filterwarnings("ignore")

def load_rag_pipeline(name = "faiss_index"):
    # Load the index
    db = FAISS.load_local(name, OpenAIEmbeddings())
    # Expose this index in a retriever interface
    retriever = db.as_retriever()
    # Create a chain to answer questions
    return RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )


# file_path = ans['source_documents'][0].metadata['source']
# page_no = ans['source_documents'][0].metadata['page']

def extract_photo(file_path,page_no):
    pdf_file = fitz.open(file_path)
    # iterate over PDF pages 
    page = pdf_file[page_no]
    images_paths = []
    
    for image_index, img in enumerate(page.get_images(), start=1): 

        # get the XREF of the image 
        xref = img[0] 

        # extract the image bytes 
        base_image = pdf_file.extract_image(xref) 
        image_bytes = base_image["image"] 

        # get the image extension 
        image_ext = base_image["ext"] 
        
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        name = os.path.splitext(os.path.basename(file_name))[0]
        # load it to PIL
        image = Image.open(io.BytesIO(image_bytes))
        if not os.path.exists("./images"):
            os.mkdir("./images")
        image_path = os.path.join("./images", "image_"+ name + str(image_index) + str(page_no) +".png")
        image.save(image_path, image_ext)
        image_load_path = os.path.join("./images", "image_"+ name +".png")
        images_paths.append(image_load_path)
    return images_paths

def get_images_from_pdf(response,pdf_path = None):
    
    for i,doc in enumerate(response['source_documents']):

        file_path = response['source_documents'][i].metadata['source']
        page_no = response['source_documents'][i].metadata['page']

        if pdf_path:
            image_paths = extract_photo(pdf_path,page_no)
        else:
            image_paths = extract_photo(file_path,page_no)
        
        if len(image_paths) > 2:
            return image_paths
    return None

def answer_question(question):
    context = "You are content creator, I need a script of 500 plus words for creating a youtube video. The content request by the user is on thr topic {0}"
    answer = qa(context.format(question))
    return answer




if __name__ == "__main__":
    print("Building RAG pipeline...")
    start = time.time()
    # Load the RAG pipeline
    qa = load_rag_pipeline()
    print("Building took {0}s".format(time.time()-start))
    print("Done building RAG pipeline")
    print("Answering question...")
    start = time.time()

    flag = True
    while flag:
        query = input("Enter a video content: ")
        if query == "exit":
            flag = False
            continue
        # Answer a question
        ans = answer_question(query)


        print(ans['result'])

    #doc_res = report_to_doctor(ans['result'])

    #print(doc_res.choices[0].message.content)
    print("Answering took {0}s".format(time.time()-start))
    print("Done answering question")

        

