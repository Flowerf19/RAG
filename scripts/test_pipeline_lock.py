from threading import Thread
from pipeline.retrieval.retrieval_orchestrator import fetch_retrieval

def call(i):
    print(f"call {i}")
    r = fetch_retrieval("test duplicate init", top_k=1, embedder_type="ollama")
    print(f"done {i}")

if __name__ == '__main__':
    t1 = Thread(target=call, args=(1,))
    t2 = Thread(target=call, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
