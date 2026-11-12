
import os

from flask import Flask, request, jsonify, render_template_string

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)

# ============================================================
# GEMINI API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")

# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY
)

# ============================================================
# KT GUIDE
# ============================================================

kt_guide_content = """
Welcome to InnovateCorp! This Knowledge Transfer (KT) guide is designed to help new employees navigate their initial weeks and understand key aspects of our operations. Our core values are Innovation, Collaboration, and Customer Focus.

Team Structure:
You will be joining the Project Alpha team, reporting to Sarah Chen, the Senior Project Manager. Your direct teammates include David Lee (Lead Developer), Maria Rodriguez (UI/UX Designer), and Tom Jackson (QA Engineer).

Team meetings are held every Monday at 10 AM in Conference Room 3, and daily stand-ups are at 9:30 AM via Google Meet.

Key Tools & Software:
For project management, we use Jira for task tracking and Confluence for documentation. Our primary communication tool is Slack for instant messaging and Google Workspace for email and calendars. Development work is primarily done using Python and JavaScript, with code hosted on GitHub. Access to these tools will be granted within your first three days.

Onboarding Process:
Your first week will focus on setup and introductions. You'll receive your laptop and login credentials on day one. HR will conduct an orientation session on Tuesday covering company policies, benefits, and payroll. You'll have one-on-one meetings with your team members throughout the week. By the end of your second week, you should have access to all necessary systems and have completed mandatory compliance training modules.

Important Resources:
The company's internal knowledge base can be found at internal.innovatecorp.com/kb. This includes FAQs, best practices, and troubleshooting guides. For IT support, please submit a ticket via support.innovatecorp.com or call extension 5555. Health and wellness benefits information is available on the HR portal.

Culture & Expectations:
InnovateCorp encourages a proactive and collaborative environment. We value open communication and continuous learning. Don't hesitate to ask questions; your team is here to support your growth. Performance reviews are conducted quarterly, and professional development courses are available through our InnovateLearn platform.
"""

# ============================================================
# DOCUMENT + CHUNKS
# ============================================================

kt_documents = [
    Document(page_content=kt_guide_content)
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

kt_chunks = splitter.split_documents(kt_documents)

# ============================================================
# EMBEDDINGS + FAISS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

vector_store = FAISS.from_documents(
    kt_chunks,
    embeddings
)

# ============================================================
# TEXT HELPER
# ============================================================

def text_of(response):
    content = getattr(response, "content", response)

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))

        return " ".join(parts).strip()

    return str(content).strip()


# ============================================================
# RELEVANCE CHECK
# ============================================================

KT_TOPICS = """
InnovateCorp, Project Alpha, Sarah Chen, David Lee,
Maria Rodriguez, Tom Jackson, team structure,
reporting manager, meetings, daily stand-ups,
Jira, Confluence, Slack, Google Workspace,
Python, JavaScript, GitHub, onboarding,
laptop, login credentials, HR orientation,
company policies, benefits, payroll,
one-on-one meetings, compliance training,
internal knowledge base, IT support, HR portal,
company culture, performance reviews,
professional development, InnovateLearn
"""


def is_kt_relevant(query):

    classification_prompt = f"""
Classify the user's question as exactly one:

RELEVANT
IRRELEVANT

The question is RELEVANT only if it is related
to the InnovateCorp KT Guide.

KT Guide topics:
{KT_TOPICS}

General knowledge, sports, politics, jokes,
unrelated programming questions, and unrelated
personal questions are IRRELEVANT.

User question:
{query}

Return ONLY:
RELEVANT
or
IRRELEVANT
"""

    response = llm.invoke(classification_prompt)

    result = text_of(response).upper()

    return result.startswith("RELEVANT")


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_kt_context(query):

    retrieved_docs = vector_store.similarity_search(
        query,
        k=3
    )

    if not retrieved_docs:
        return ""

    return "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


# ============================================================
# FINAL ANSWER
# ============================================================

def answer_kt_question(query):

    query = query.strip()

    if not query:
        return {
            "answer": "Please enter a question.",
            "relevant": False,
            "tool_used": False
        }

    # IMPORTANT:
    # Relevance is checked BEFORE retrieval.
    relevant = is_kt_relevant(query)

    # Irrelevant question:
    # NO retrieval happens.
    if not relevant:
        return {
            "answer": (
                "This question is outside the InnovateCorp "
                "KT Guide. Please ask about onboarding, "
                "Project Alpha, team roles, company tools, "
                "resources, processes, or company culture."
            ),
            "relevant": False,
            "tool_used": False
        }

    # Relevant question:
    # Retrieval happens here.
    context = retrieve_kt_context(query)

    answer_prompt = f"""
You are an InnovateCorp KT Guide assistant.

Answer the user's question using ONLY the KT Guide
context below.

Do not invent information.

If the answer is not available in the context, say:

"That information is not available in the KT Guide."

KT GUIDE CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    response = llm.invoke(answer_prompt)

    return {
        "answer": text_of(response),
        "relevant": True,
        "tool_used": True
    }


# ============================================================
# WEB PAGE
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>InnovateCorp KT Guide</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f7fa;
        }

        .container {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
        }

        textarea {
            width: 100%;
            height: 100px;
            padding: 12px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 8px;
            resize: vertical;
        }

        button {
            margin-top: 12px;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        #answer {
            margin-top: 25px;
            padding: 20px;
            background: #f1f5f9;
            border-radius: 8px;
            white-space: pre-wrap;
        }
    </style>
</head>

<body>

<div class="container">

<h1>InnovateCorp KT Guide</h1>

<p>
Ask questions about onboarding, Project Alpha,
team members, tools, company resources and culture.
</p>

<textarea id="question"
placeholder="Enter your question..."></textarea>

<br>

<button onclick="askQuestion()">Ask Question</button>

<div id="answer">
Answer will appear here.
</div>

</div>

<script>

async function askQuestion() {

    const question =
        document.getElementById("question").value;

    const answer =
        document.getElementById("answer");

    answer.innerText = "Thinking...";

    const response = await fetch("/ask", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            question: question
        })

    });

    const data = await response.json();

    answer.innerText = data.answer;
}

</script>

</body>
</html>
"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    question = data.get("question", "")

    result = answer_kt_question(question)

    return jsonify(result)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
