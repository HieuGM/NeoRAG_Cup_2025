import pandas as pd
from reranker import Reranker
from typing import List, Dict, Optional, Tuple
import numpy as np
import os
from openai import OpenAI
import time


# Khởi tạo các thành phần
reranker = Reranker()

llm_model = 'gpt-4.1-nano'
# Chỉ embedding, reranking 1 lần
_embedding_cache = {}
_rerank_cache = {}

def get_user_embedding(query, embedding):
    if query in _embedding_cache:
        return _embedding_cache[query]
    user_embedding = embedding.encode(query)
    _embedding_cache[query] = user_embedding
    return user_embedding

def apply_reranking(query, embedding, vector_db, reranker, k):
    """
    Áp dụng reranking cho kết quả tìm kiếm
    """
    user_embedding = get_user_embedding(query, embedding)
    init_limit = 2 * k
    # Kiểm tra xem có lặp lại rerank không
    cache_key = (query, k)
    if cache_key in _rerank_cache:
        return _rerank_cache[cache_key]
    #retrieval initial
    results = vector_db.query("information", user_embedding, limit=init_limit)
    if reranker and results:
        #Tạo danh sách để đưa vào rerank
        passages = [res['information'] for res in results]
        #rerank các passages giảm dần theo độ liên quan
        ranked_scores, ranked_passages = reranker(query, passages)
        reranked_results = []
        for rp in ranked_passages[:k]:
            for res in results:
                if res['information'] == rp:
                    reranked_results.append(res)
                    break
        _rerank_cache[cache_key] = reranked_results
        return reranked_results
    topk = results[:k]
    _rerank_cache[cache_key] = topk
    return topk

hitk = []
reck = []
def hit_k(file_clb_proptit, file_train_data_proptit, embedding, vector_db, k):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train_data_proptit)
    
    hits = 0
    total_queries = len(df_train)

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        
        print(f"Đang xử lý query {index} tại k = {k}, hàm hit_k")
        results = apply_reranking(query, embedding, vector_db, reranker, k=k)
        
        # Lấy top-k cuối cùng
        retrieved_docs = [int(result['title'].split()[-1]) for result in results[:k] if 'title' in result]

        # Parse ground truth
        ground_truth_docs = []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))

        # Đánh giá
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits += 1
    hitk.append(hits / total_queries if total_queries > 0 else 0)
    print(hitk[0])
    return hits / total_queries if total_queries > 0 else 0

# Hàm recall@k
def recall_k(file_clb_proptit, file_train, embedding, vector_db, k):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    ans = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        print(f"Đang xử lý query {index} tại k = {k}, hàm recall_k")
        ground_truth_doc = row['Ground truth document']
        
        results = apply_reranking(query, embedding, vector_db, reranker, k)
        
        # Lấy top-k cuối cùng
        retrieved_docs = [int(result['title'].split()[-1]) for result in results[:k] if 'title' in result]
        # Lấy danh sách tài liệu được truy suất
        ground_truth_docs = []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
            
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits = len([doc for doc in ground_truth_docs if doc in retrieved_docs])
        ans += hits / len(ground_truth_docs) 
    reck.append(ans / len(df_train))
    print(reck[0])
    return ans / len(df_train)

prek = []
# Hàm precision@k
def precision_k(file_clb_proptit, file_train, embedding, vector_db, k):
    """
    Tính Precision@k với query rewriting để tăng độ chính xác
    """
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    ans = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        print(f"Đang xử lý query {index} tại k = {k}, hàm precision_k")
        
        # Sử dụng query rewriting để tăng độ chính xác
        results = apply_reranking(query, embedding, vector_db, reranker, k=k)
        
        # Lấy top-k cuối cùng
        retrieved_docs = [int(result['title'].split()[-1]) for result in results[:k] if 'title' in result]
        ground_truth_docs = []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
            
        # Kiểm tra xem có ít nhất một tài liệu đúng trong kết quả tìm kiếm
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits = len([doc for doc in retrieved_docs if doc in ground_truth_docs])
        ans += hits / k 
    prek.append(ans / len(df_train))
    print(prek[0])
    return ans / len(df_train)


# Hàm f1@k
def f1_k(file_clb_proptit, file_train, embedding, vector_db, k):
    precision = prek[0] if prek else 0
    recall = reck[0] if reck else 0
    if precision + recall == 0:
        return 0
    print(2 * (precision * recall) / (precision + recall))
    return 2 * (precision * recall) / (precision + recall)

# Hàm MAP@k

def map_k(file_clb_proptit, file_train, embedding, vector_db, k):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_map = 0

    for index, row in df_train.iterrows():
        hits = 0
        ap = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        print(f"Đang xử lý query {index} tại k = {k}, hàm map_k")
        # Tạo embedding cho câu hỏi của người dùng
        results = apply_reranking(query, embedding, vector_db, reranker, k)

        # Lấy danh sách tài liệu được truy suất
        retrieved_docs = [int(result['title'].split(' ')[-1]) for result in results if 'title' in result]
        # print(f"Retrieved documents: {retrieved_docs}")
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        # print("Ground truth documents:", ground_truth_docs)
        
        # Tính MAP cho 1 truy vấn
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_docs:
                hits += 1
                ap += hits / (i + 1)
        if hits > 0:
            ap /= hits
        # print(f"Average Precision for this query: {ap}")
        total_map += ap 
    print(total_map / len(df_train))
    return total_map / len(df_train)

# Hàm MRR@k
def mrr_k(file_clb_proptit, file_train, embedding, vector_db, k):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_mrr = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        print(f"Đang xử lý query {index} tại k = {k}, hàm mrr_k")
        
        #Reranking
        results = apply_reranking(query, embedding, vector_db, reranker, k=k)

        # Lấy danh sách tài liệu được truy suất
        retrieved_docs = [int(result['title'].split(' ')[-1]) for result in results if 'title' in result]
        # print(f"Retrieved documents: {retrieved_docs}")
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        # print("Ground truth documents:", ground_truth_docs)
        
        # Tính MRR cho 1 truy vấn
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_docs:
                total_mrr += 1 / (i + 1)
                break
    print(total_mrr / len(df_train) if len(df_train) > 0 else 0)
    return total_mrr / len(df_train) if len(df_train) > 0 else 0

# Hàm NDCG@k
import numpy as np
def dcg_at_k(relevances, k):
    relevances = np.array(relevances)[:k]
    return np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))

def ndcg_at_k(relevances, k):
    relevances = np.array(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    
    dcg = dcg_at_k(relevances, k)
    dcg_max = dcg_at_k(sorted(relevances, reverse=True), k)
    
    if dcg_max == 0:
        return 0.0
    
    return dcg / dcg_max

def similarity(embedding1, embedding2):
    # Giả sử ta có một hàm để tính độ tương đồng giữa hai embedding
    # Ở đây ta sẽ sử dụng cosine similarity, chuẩn hóa để score nước về khoảng [0, 1]
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_sim = dot_product / (norm1 * norm2)
    return (cos_sim + 1) / 2


def ndcg_k(file_clb_proptit, file_train, embedding, vector_db, k):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_ndcg = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        print(f"Đang xử lý query {index} tại k = {k}, hàm ndcg_k")
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = get_user_embedding(query, embedding)

        # Reranking
        results = apply_reranking(query, embedding, vector_db, reranker, k)

        retrieved_docs = [int(result['title'].split(' ')[-1]) for result in results]

        ground_truth_docs = []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))


        # Nếu điểm tương đồng > 0.9 thì gán 3, nếu > 0.7 thì gán 2, nếu > 0.5 thì gán 1, còn lại thì gán 0 
        relevances = []
        for doc in retrieved_docs:
            if doc in ground_truth_docs:
                # Giả sử ta có một hàm để tính độ tương đồng giữa câu hỏi và tài liệu, doc là số thứ tự của tài liệu trong file CLB_PROPTIT.csv
                doc_result = [res for res in results if int(res['title'].split(' ')[-1]) == doc][0]
                doc_embedding = embedding.encode(doc_result['information'])
                similarity_score = similarity(user_embedding, doc_embedding)
                if similarity_score > 0.9:
                    relevances.append(3)
                elif similarity_score > 0.7:
                    relevances.append(2)
                elif similarity_score > 0.5:
                    relevances.append(1)
                else:
                    relevances.append(0)
            else:
                relevances.append(0)
        ndcg = ndcg_at_k(relevances, k)
        total_ndcg += ndcg
    print(total_ndcg / len(df_train) if len(df_train) > 0 else 0)
    return total_ndcg / len(df_train) if len(df_train) > 0 else 0

# Hàm Context Precision@k (LLM Judged)

def context_precision_k(file_clb_proptit, file_train, embedding, vector_db, k):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    total_precision = 0

    for index, row in df_train.iterrows():
        # TODO: Tạo ra LLM Answer, các em hãy tự viết phần system prompt
        messages = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
            }
        ]
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        print(f"Đang xử lý query {index} tại k = {k}, hàm cp_k")

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = apply_reranking(query, embedding, vector_db, reranker, k)

        # TODO: viết câu query của người dùng (bao gồm document retrieval và câu query)
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([res["information"] for res in results])
        context = context + "\n\nCâu hỏi: " + query

        # Thêm context vào messages
        messages.append({
            "role": "user",
            "content": context
        })
        # Gọi API để lấy câu trả lời
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages
            )
            reply = response.choices[0].message.content.strip()
        except Exception:
            reply = ""

        # Đẩy các đoạn văn được retrieved và câu trả lời của LLM vào một LLM Judged context với prompt system
        # LLM Judged context
        for result in results:
            # NOTE: Các em có thể chỉnh messages_judged nếu muốn
            messages_judged = [
                {
                    "role": "system",
                    "content": """Bạn là một trợ lý AI chuyên đánh giá độ chính xác của các câu trả lời dựa trên ngữ cảnh được cung cấp. 
                    Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trả lời từ một mô hình AI. 
                    Nhiệm vụ của bạn là đánh giá câu trả lời dựa trên ngữ cảnh và câu hỏi. 
                    Nếu ngữ cảnh và câu hỏi cung cấp đủ thông tin hoặc chỉ cần một phần thông tin để trả lời câu hỏi, hãy đánh giá câu trả lời là 1. Nếu không, hãy đánh giá là 0. 
                    Hãy đọc thật kĩ ngữ cảnh, chỉ cần ngữ cảnh có một phần thông tin để trả lời cho một phần của câu hỏi thì cũng đánh giá là 1. Nếu ngữ cảnh không liên quan gì đến câu hỏi, hãy đánh giá là 0. 
                    LƯU Ý: Chỉ trả lời 1 hoặc 0, không giải thích gì thêm, không thừa bất cứ ký tự nào khác."""
                }
            ]
            messages_judged.append({
                "role": "user",
                "content": f"Ngữ cảnh: {result['information']}\n\nCâu hỏi: {query}\n\nCâu trả lời: {reply}"
            })
            # Gọi API đến LLM Judged
            try:
                judged_response = client.chat.completions.create(
                    model=llm_model,
                    messages=messages_judged
                )
                judged_reply = judged_response.choices[0].message.content.strip()
            except Exception:
                judged_reply = "0"
            if judged_reply == "1":
                hits += 1
        precision = hits / k if k > 0 else 0
        total_precision += precision
    print(total_precision / len(df_train) if len(df_train) > 0 else 0)
    return total_precision / len(df_train) if len(df_train) > 0 else 0


# Hàm Context Recall@k (LLM Judged)
def context_recall_k(file_clb_proptit, file_train, embedding, vector_db, k):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_recall = 0

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        print(f"Đang xử lý query {index} tại k = {k}, hàm cr_k")

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = apply_reranking(query, embedding, vector_db, reranker, k=k)
        reply = row['Ground truth answer']
        

        # NOTE: Các em có thể thay đổi messages_judged nếu muốn 
        for result in results:
            messages_judged = [
                {
                    "role": "system",
                    "content": """Bạn là một trợ lý AI chuyên đánh giá độ chính xác của các câu trả lời dựa trên ngữ cảnh được cung cấp. 
                    Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trả lời đã được chuyên gia trả lời cho câu hỏi (đây là câu trả lời chính xác). 
                    Nhiệm vụ của bạn là đánh giá ngữ cảnh dựa trên câu hỏi và câu trả lời. 
                    Nếu ngữ cảnh và câu hỏi cung cấp đủ thông tin hoặc chỉ cần một phần thông tin để trả lời câu hỏi, hãy đánh giá ngữ cảnh là 1. Nếu không, hãy đánh giá là 0. Hãy đọc thật kĩ ngữ cảnh, chỉ cần ngữ cảnh có một phần thông tin để trả lời cho một phần của câu hỏi thì cũng đánh giá là 1. Nếu ngữ cảnh không liên quan gì đến câu hỏi, hãy đánh giá là 0. 
                    LƯU Ý: Chỉ trả lời 1 hoặc 0, không giải thích gì thêm."""
                }
            ]
            messages_judged.append({
                "role": "user",
                "content": f"Ngữ cảnh: {result['information']}\n\nCâu hỏi: {query}\n\nCâu trả lời: {reply}"
            })
            # Gọi API đến LLM Judged
            try:
                judged_response = client.chat.completions.create(
                    model=llm_model,
                    messages=messages_judged,
                )
                judged_reply = judged_response.choices[0].message.content.strip()
            except Exception:
                judged_reply = "0"
            if judged_reply == "1":
                hits += 1
        recall = hits / k if k > 0 else 0
        total_recall += recall
    print(total_recall / len(df_train) if len(df_train) > 0 else 0)
    return total_recall / len(df_train) if len(df_train) > 0 else 0

# Hàm Context Entities Recall@k (LLM Judged)
def context_entities_recall_k(file_clb_proptit, file_train, embedding, vector_db, k):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_recall = 0
    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        print(f"Đang xử lý query {index} tại k = {k}, hàm cer_k")
        # Tạo embedding cho câu hỏi của người dùng
        user_embedding = get_user_embedding(query, embedding)

        # Tìm kiếm thông tin liên quan trong cơ sở dữ liệu
        results = apply_reranking(query, embedding, vector_db, reranker, k)
        # results = vector_db.query("information", user_embedding, limit=k)
        reply = row['Ground truth answer']
        # Trích xuất các thực thể từ Ground truth answer bằng LLM
        # NOTE: Các em có thể thay đổi messages_entities nếu muốn
        messages_entities = [
            {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên trích xuất các thực thể từ câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là trích xuất các thực thể từ câu trả lời đó. Các thực thể có thể là tên người, địa điểm, tổ chức, sự kiện, v.v. Hãy trả lời dưới dạng một danh sách các thực thể.
                Ví dụ:
                Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
                ["ngành khác", "CLB", "CNTT", "mảng]
                Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
                ["Câu lạc bộ Lập Trình PTIT (Programming PTIT)", "PROPTIT", "9/10/2011", "Chia sẻ để cùng nhau phát triển", "sinh viên", "Học viện", "Lập Trình PTIT - Lập trình từ trái tim"]"""
            }
        ]
        # NOTE: Các em có thể thay đổi content nếu muốn
        messages_entities.append({
            "role": "user",
            "content": f"Câu trả lời: {reply}"
        })
        # Gọi  API để trích xuất các thực thể
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            entities_response = client.chat.completions.create(
                model=llm_model,
                messages=messages_entities
            )
            entities = entities_response.choices[0].message.content.strip().split("\n")
            entities = entities[0] # "["ngành khác", "CLB", "CNTT", "mảng]" -> ["ngành khác", "CLB", "CNTT", "mảng"]
            entities = eval(entities) if entities else []  # Chuyển đổi chuỗi thành danh sách
        except Exception:
            entities = []
        
        tmp = len(entities)
        for result in results:
            context = result['information']
            for entity in entities:
                if entity.strip() in context:
                    hits += 1
                    entities.remove(entity.strip())
        total_recall += hits / tmp if tmp > 0 else 0
    print(total_recall / len(df_train) if len(df_train) > 0 else 0)
    return total_recall / len(df_train) if len(df_train) > 0 else 0



# Hàm tính toán tất cả metrics liên quan đến Retrieval

def calculate_metrics_retrieval(file_clb_proptit, file_train , embedding, vector_db, train):
    # Tạo ra 1 bảng csv, cột thứ nhất là K value, các cột còn lại là metrics. Sẽ có 3 hàng tương trưng với k = 3, 5, 7
    
    k_values = [7]
    metrics = {
        "K": [],
        "hit@k": [],
        "recall@k": [],
        "precision@k": [],
        "f1@k": [],
        "map@k": [],
        "mrr@k": [],
        "ndcg@k": [],
        "context_precision@k": [],
        "context_recall@k": [],
        "context_entities_recall@k": []
    }
    # Lưu 2 chữ số thập phân cho các metrics
    # Precompute per-query caches for max k to reuse across metrics
    max_k = max(k_values)

    # Compute metrics per k using cached apply_reranking via internal caches
    for k in k_values:
        metrics["K"].append(k)
        metrics["hit@k"].append(round(hit_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["recall@k"].append(round(recall_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["precision@k"].append(round(precision_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["f1@k"].append(round(f1_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["map@k"].append(round(map_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["mrr@k"].append(round(mrr_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["ndcg@k"].append(round(ndcg_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["context_precision@k"].append(round(context_precision_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["context_recall@k"].append(round(context_recall_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["context_entities_recall@k"].append(round(context_entities_recall_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
    # Chuyển đổi metrics thành DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
    if train:
        metrics_df.to_csv("metrics_retrieval_train.csv", index=False)
    else:
        metrics_df.to_csv("metrics_retrieval_test1.csv", index=False)
    return metrics_df

# Các hàm đánh giá LLM Answer

# Hàm String Presence

def string_presence_k(file_clb_proptit: str, file_train: str, embedding, vector_db, k):
    # Khởi tạo OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Đọc dữ liệu
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_presence = 0.0
    total_queries = len(df_train)
    
    if total_queries == 0:
        return 0.0
    
    # Cache để tránh gọi API trùng lặp
    _response_cache = {}
    _entities_cache = {}
    
    for index, row in df_train.iterrows():
        try:
            query = row['Query']
            reply = row['Ground truth answer']
            print(f'Query {index}, k = {k}, hàm spk')
            
            # Tạo cache key cho query
            cache_key = f"{query}_{k}"
            
            # Kiểm tra cache cho response
            if cache_key in _response_cache:
                response = _response_cache[cache_key]
            else:
                # Tạo embedding cho câu hỏi
                user_embedding = get_user_embedding(query, embedding)
                
                # Tìm kiếm thông tin liên quan
                results = apply_reranking(query, embedding, vector_db, reranker, k=k)
                
                # Tạo context từ kết quả tìm kiếm
                context = "Content từ các tài liệu liên quan:\n"
                context += "\n".join([result["information"] for result in results])
                
                # Tạo messages cho API call
                messages = [
                    {
                        "role": "system",
                        "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

Nguyên tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời
2. Trình bày câu trả lời rõ ràng, dễ hiểu
3. Tuyệt đối không suy đoán hoặc bịa thông tin
4. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán
5. Tập trung vào câu hỏi của người dùng để trả lời chính xác nhất"""
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nCâu hỏi: {query}"
                    }
                ]
                
                # Gọi API để lấy câu trả lời
                response_obj = client.chat.completions.create(
                    model=llm_model,
                    messages=messages
                )
                response = response_obj.choices[0].message.content.strip()
                _response_cache[cache_key] = response
                time.sleep(30)
            
            # Kiểm tra cache cho entities
            if reply in _entities_cache:
                entities = _entities_cache[reply]
            else:
                # Trích xuất các thực thể từ ground truth answer
                messages_entities = [
                    {
                        "role": "system",
                        "content": """Bạn là một trợ lý AI chuyên trích xuất các thực thể từ câu trả lời. 
Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là trích xuất các thực thể từ câu trả lời đó. 
Các thực thể có thể là tên người, địa điểm, tổ chức, sự kiện, v.v. 
Hãy trả lời dưới dạng một danh sách các thực thể.

Ví dụ:
Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển.
["ngành khác", "CLB", "CNTT"]

Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011.
["Câu lạc bộ Lập Trình PTIT", "PROPTIT", "9/10/2011"]"""
                    },
                    {
                        "role": "user",
                        "content": f"Câu trả lời: {reply}"
                    }
                ]
                
                # Gọi API để trích xuất các thực thể
                entities_response = client.chat.completions.create(
                    model=llm_model,
                    messages=messages_entities
                )
                
                entities_text = entities_response.choices[0].message.content.strip()
                
                # Xử lý kết quả entities
                try:
                    # Tìm dòng chứa danh sách entities
                    entities_lines = entities_text.split("\n")
                    entities_line = None
                    for line in entities_lines:
                        if line.strip().startswith("[") and line.strip().endswith("]"):
                            entities_line = line.strip()
                            break
                    
                    if entities_line:
                        # Sử dụng eval để chuyển đổi chuỗi thành danh sách (an toàn trong context này)
                        entities = eval(entities_line) if entities_line else []
                    else:
                        entities = []
                except (SyntaxError, ValueError):
                    entities = []
                
                _entities_cache[reply] = entities
                time.sleep(30)  # Thêm delay để tránh vượt quá giới hạn tốc độ API
                
            
            # Tính toán hits
            hits = 0.0
            if entities and len(entities) > 0:
                for entity in entities:
                    if entity.strip() and entity.strip() in response:
                        hits += 1.0
                hits /= len(entities)
            
            total_presence += hits
            
        except Exception as e:
            print(f"Lỗi xử lý query {index}: {str(e)}")
            continue
    print(total_presence/total_queries)
    return total_presence / total_queries


 

# Hàm Rouge-L

from rouge import Rouge
def rouge_l_k(file_clb_proptit, file_train, embedding, vector_db, k):
    # Khởi tạo OpenAI client và Rouge
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    rouge = Rouge()
    
    # Đọc dữ liệu
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)
    
    total_rouge_l = 0.0
    total_queries = len(df_train)
    
    if total_queries == 0:
        return 0.0
    
    # Cache để tránh gọi API trùng lặp
    _response_cache = {}
    
    for index, row in df_train.iterrows():
        try:
            query = row['Query']
            reply = row['Ground truth answer']
            print(f'Query {index}, k = {k}, hàm rlk')
            
            # Tạo cache key cho query
            cache_key = f"{query}_{k}"
            
            # Kiểm tra cache cho response
            if cache_key in _response_cache:
                response = _response_cache[cache_key]
            else:
                # Tạo embedding cho câu hỏi
                user_embedding = get_user_embedding(query, embedding)
                
                # Tìm kiếm thông tin liên quan
                results = apply_reranking(query, embedding, vector_db, reranker, k=k)
                
                # Tạo context từ kết quả tìm kiếm
                context = "Content từ các tài liệu liên quan:\n"
                context += "\n".join([result["information"] for result in results])
                
                # Tạo messages cho API call
                messages = [
                    {
                        "role": "system",
                        "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

Nguyên tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời
2. Trình bày câu trả lời rõ ràng, dễ hiểu
3. Tuyệt đối không suy đoán hoặc bịa thông tin
4. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán
5. Tập trung vào câu hỏi của người dùng để trả lời chính xác nhất"""
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nCâu hỏi: {query}"
                    }
                ]
                
                # Gọi API để lấy câu trả lời
                response_obj = client.chat.completions.create(
                    model=llm_model,
                    messages=messages
                )
                response = response_obj.choices[0].message.content.strip()
                _response_cache[cache_key] = response
                time.sleep(21)
            
            # Tính toán Rouge-L score
            try:
                scores = rouge.get_scores(response, reply)
                rouge_l_score = scores[0]['rouge-l']['f']
                total_rouge_l += rouge_l_score
            except Exception as e:
                print(f"Lỗi tính Rouge-L cho query {index}: {str(e)}")
                continue
                
        except Exception as e:
            print(f"Lỗi xử lý query {index}: {str(e)}")
            continue
    print(total_rouge_l / total_queries)
    return total_rouge_l / total_queries

# Hàm BLEU-4
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu_4_k(file_clb_proptit, file_train, embedding, vector_db, k):
    """
    Tính toán điểm số BLEU-4 giữa câu trả lời của LLM và ground truth answer.
    """
    # Khởi tạo OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Đọc dữ liệu
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_bleu_4 = 0.0
    total_queries = len(df_train)
    smoothing_function = SmoothingFunction().method1
    
    if total_queries == 0:
        return 0.0
    
    # Cache để tránh gọi API trùng lặp
    _response_cache = {}

    for index, row in df_train.iterrows():
        try:
            query = row['Query']
            reply = row['Ground truth answer']
            print(f'Query {index}, k = {k}, hàm b4k')
            
            # Tạo cache key cho query
            cache_key = f"{query}_{k}"
            
            # Kiểm tra cache cho response
            if cache_key in _response_cache:
                response = _response_cache[cache_key]
            else:
                # Tạo embedding cho câu hỏi
                user_embedding = get_user_embedding(query, embedding)

                # Tìm kiếm thông tin liên quan
                results = apply_reranking(query, embedding, vector_db, reranker, k=k)
                
                # Tạo context từ kết quả tìm kiếm
                context = "Content từ các tài liệu liên quan:\n"
                context += "\n".join([result["information"] for result in results])
                
                # Tạo messages cho API call
                messages = [
                    {
                        "role": "system",
                        "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

Nguyên tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời
2. Trình bày câu trả lời rõ ràng, dễ hiểu
3. Tuyệt đối không suy đoán hoặc bịa thông tin
4. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán
5. Tập trung vào câu hỏi của người dùng để trả lời chính xác nhất"""
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nCâu hỏi: {query}"
                    }
                ]

                # Gọi API để lấy câu trả lời
                response_obj = client.chat.completions.create(
                    model=llm_model,
                    messages=messages
                )
                response = response_obj.choices[0].message.content.strip()
                _response_cache[cache_key] = response
                time.sleep(21)
            
            # Tính toán BLEU-4 score
            try:
                reference = reply.split()
                candidate = response.split()
                bleu_4 = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)
                total_bleu_4 += bleu_4
            except Exception as e:
                print(f"Lỗi tính BLEU-4 cho query {index}: {str(e)}")
                continue
                
        except Exception as e:
            print(f"Lỗi xử lý query {index}: {str(e)}")
            continue
    print(total_bleu_4 / total_queries)
    return total_bleu_4 / total_queries

# Hàm Groundedness (LLM Answer - Hallucination Detection)\

def groundedness_k(file_clb_proptit, file_train, embedding, vector_db, k):
    """
    Tính toán điểm số Groundedness để phát hiện hallucination trong câu trả lời của LLM.
    """
    # Khởi tạo OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Đọc dữ liệu
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_groundedness = 0.0
    total_queries = len(df_train)
    
    if total_queries == 0:
        return 0.0
    
    # Cache để tránh gọi API trùng lặp
    _response_cache = {}

    for index, row in df_train.iterrows():
        try:
            hits = 0
            cnt = 0
            query = row['Query']
            print(f'Query {index}, k = {k}, hàm gk')
            reply = row['Ground truth answer']
            
            # Tạo cache key cho query
            cache_key = f"{query}_{k}"
            
            # Kiểm tra cache cho response
            if cache_key in _response_cache:
                response = _response_cache[cache_key]
            else:
                # Tạo embedding cho câu hỏi
                user_embedding = get_user_embedding(query, embedding)

                # Tìm kiếm thông tin liên quan
                results = apply_reranking(query, embedding, vector_db, reranker, k=k)
                
                # Tạo context từ kết quả tìm kiếm
                context = "Content từ các tài liệu liên quan:\n"
                context += "\n".join([result["information"] for result in results])
                
                # Tạo messages cho API call
                messages = [
                {
                "role": "system",
                "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nCâu hỏi: {query}"
                    }
                ]
                
                # Gọi API để lấy câu trả lời
                response_obj = client.chat.completions.create(
                    model=llm_model,
                    messages=messages
                )
                response = response_obj.choices[0].message.content.strip()
                _response_cache[cache_key] = response
                time.sleep(21)
            
            # Tách response thành các câu
            sentences = response.split('. ')
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Tạo prompt để kiểm tra tính groundedness của câu
                messages_groundedness = [
                    {
                        "role": "system",
                        "content": """Bạn là một chuyên gia đánh giá Groundedness trong hệ thống RAG, có nhiệm vụ phân loại từng câu của câu trả lời dựa trên ngữ cảnh đã cho.
                        Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trong phần trả lời từ mô hình AI. Nhiệm vụ của bạn là đánh giá câu trả lời đó dựa trên ngữ cảnh và câu hỏi.
                        Input:
                        Question: Câu hỏi của người dùng
                        Contexts: Một hoặc nhiều đoạn văn bản được truy xuất
                        Answer: Chỉ một câu trong đoạn văn bản LLM sinh ra
                        Bạn hãy đánh giá dựa trên các nhãn sau: 
                        supported: Nội dung câu được ngữ cảnh hỗ trợ hoặc suy ra trực tiếp.
                        unsupported: Nội dung câu không được ngữ cảnh hỗ trợ, và không thể suy ra từ đó.
                        contradictory: Nội dung câu trái ngược hoặc mâu thuẫn với ngữ cảnh.
                        no_rad: Câu không yêu cầu kiểm tra thực tế (ví dụ: câu chào hỏi, ý kiến cá nhân, câu hỏi tu từ, disclaimers).
                        Hãy trả lời bằng một trong các nhãn trên, không giải thích gì thêm. Chỉ trả lời một từ duy nhất là nhãn đó.
                        Ví dụ:
                        Question: Bạn có thể cho tôi biết về lịch sử của Câu lạc bộ Lập trình ProPTIT không?
                        Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                        Answer: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011.
                        supported"""
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nContexts: {context}\n\nAnswer: {sentence.strip()}"
                    }
                ]
                
                # Gọi API để đánh giá groundedness
                try:
                    groundedness_response = client.chat.completions.create(
                        model=llm_model,
                        messages=messages_groundedness
                    )
                    groundedness_reply = groundedness_response.choices[0].message.content.strip().lower()
                    
                    if groundedness_reply == "supported":
                        hits += 1
                        cnt += 1
                    elif groundedness_reply in ["unsupported", "contradictory"]:
                        cnt += 1
                    time.sleep(21)  # Thêm delay để tránh vượt quá giới hạn tốc độ API
                except Exception as e:
                    print(f"Lỗi gọi API groundedness cho câu {index}: {str(e)}")
                    continue
                    
            if cnt > 0:
                total_groundedness += hits / cnt
                
        except Exception as e:
            print(f"Lỗi xử lý query {index}: {str(e)}")
            continue
    print(total_groundedness / total_queries)
    return total_groundedness / total_queries 

# Hàm Response Relevancy (LLM Answer - Measures relevance)


def generate_related_questions(response: str, embedding) -> list:
    """
    Tạo ra các câu hỏi liên quan từ một câu trả lời.
    
    Args:
        response: Câu trả lời từ LLM
        embedding: Model embedding để tạo vector
        
    Returns:
        list: Danh sách các câu hỏi liên quan
    """
    # Khởi tạo OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Tạo system prompt
    messages_related = [
        {
            "role": "system",
            "content": """Bạn là một trợ lý AI chuyên tạo ra các câu hỏi liên quan từ một câu trả lời. 
Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là tạo ra các câu hỏi liên quan đến câu trả lời đó. 
Hãy tạo ra ít nhất 5 câu hỏi liên quan, mỗi câu hỏi nên ngắn gọn và rõ ràng. 
Trả lời dưới dạng list các câu hỏi như ở ví dụ dưới. 
LƯU Ý: Trả lời dưới dạng ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3", ...], bao gồm cả dấu ngoặc vuông.

Ví dụ:
Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011.
Output của bạn: ["CLB Lập Trình PTIT được thành lập khi nào?", "Slogan của CLB là gì?", "Mục tiêu của CLB là gì?"]

Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình.
Output của bạn: ["Ngành nào có thể tham gia CLB?", "CLB phù hợp với những ai?", "Trở ngại lớn nhất khi tham gia CLB là gì?"]"""
        },
        {
            "role": "user",
            "content": f"Câu trả lời: {response}"
        }
    ]
    
    # Gọi API để tạo ra các câu hỏi liên quan
    try:
        related_response = client.chat.completions.create(
            model=llm_model,
            messages=messages_related
        )
        related_questions_text = related_response.choices[0].message.content.strip()
        time.sleep(21)
        # Xử lý kết quả
        try:
            # Tìm dòng chứa danh sách questions
            lines = related_questions_text.split("\n")
            questions_line = None
            for line in lines:
                if line.strip().startswith("[") and line.strip().endswith("]"):
                    questions_line = line.strip()
                    break
            
            if questions_line:
                # Sử dụng eval để chuyển đổi chuỗi thành danh sách (an toàn trong context này)
                related_questions = eval(questions_line) if questions_line else []
            else:
                related_questions = []
        except (SyntaxError, ValueError):
            related_questions = []
            
        return related_questions
        
    except Exception as e:
        print(f"Lỗi tạo related questions: {str(e)}")
        return []

def response_relevancy_k(file_clb_proptit, file_train, embedding, vector_db, k):
    """
    Tính toán điểm số Response Relevancy để đo lường mức độ liên quan của câu trả lời.
    """
    # Khởi tạo OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Đọc dữ liệu
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_relevancy = 0.0
    total_queries = len(df_train)
    
    if total_queries == 0:
        return 0.0
    
    # Cache để tránh gọi API trùng lặp
    _response_cache = {}

    for index, row in df_train.iterrows():
        try:
            hits = 0.0
            query = row['Query']
            reply = row['Ground truth answer']
            print(f'Query {index}, k = {k}, hàm rrk')
            
            # Tạo cache key cho query
            cache_key = f"{query}_{k}"
            
            # Kiểm tra cache cho response
            if cache_key in _response_cache:
                response = _response_cache[cache_key]
            else:
                # Tạo embedding cho câu hỏi
                user_embedding = get_user_embedding(query, embedding)

                # Tìm kiếm thông tin liên quan
                results = apply_reranking(query, embedding, vector_db, reranker, k=k)
                
                # Tạo context từ kết quả tìm kiếm
                context = "Content từ các tài liệu liên quan:\n"
                context += "\n".join([result["information"] for result in results])
                
                # Tạo messages cho API call
                messages = [
                    {
                        "role": "system",
                        "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nCâu hỏi: {query}"
                    }
                ]
                
                # Gọi API để lấy câu trả lời
                response_obj = client.chat.completions.create(
                    model=llm_model,
                    messages=messages
                )
                response = response_obj.choices[0].message.content.strip()
                _response_cache[cache_key] = response
                time.sleep(21)

            # Dùng câu trả lời của LLM để sinh ra các câu hỏi liên quan
            related_questions = generate_related_questions(response, embedding)
            
            if related_questions and len(related_questions) > 0:
                for question in related_questions:
                    try:
                        question_embedding = get_user_embedding(question, embedding)
                        # Tính score relevancy giữa câu hỏi và query
                        score = similarity(user_embedding, question_embedding)
                        hits += score
                    except Exception as e:
                        print(f"Lỗi tính similarity cho question {question}: {str(e)}")
                        continue
                        
                total_relevancy += hits / len(related_questions)
                
        except Exception as e:
            print(f"Lỗi xử lý query {index}: {str(e)}")
            continue
    print(total_relevancy / total_queries)
    return total_relevancy / total_queries


# Hàm Noise Sensitivity (LLM Answer - Robustness to Hallucination)

def noise_sensitivity_k(file_clb_proptit, file_train, embedding, vector_db, k):
    # Khởi tạo OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Đọc dữ liệu
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_sensitivity = 0.0
    total_queries = len(df_train)
    
    if total_queries == 0:
        return 0.0
    
    # Cache để tránh gọi API trùng lặp
    _response_cache = {}

    for index, row in df_train.iterrows():
        try:
            hits = 0
            cnt = 0
            query = row['Query']
            print(f'Query {index}, k = {k}, hàm nsk')
            reply = row['Ground truth answer']
            
            # Tạo cache key cho query
            cache_key = f"{query}_{k}"
            
            # Kiểm tra cache cho response
            if cache_key in _response_cache:
                response = _response_cache[cache_key]
            else:
                # Tạo embedding cho câu hỏi
                user_embedding = get_user_embedding(query, embedding)

                # Tìm kiếm thông tin liên quan
                results = apply_reranking(query, embedding, vector_db, reranker, k=k)
                
                # Tạo context từ kết quả tìm kiếm
                context = "Content từ các tài liệu liên quan:\n"
                context += "\n".join([result["information"] for result in results])
                
                # Tạo messages cho API call
                messages = [
                    {
                        "role": "system",
                        "content": """Bạn là một trợ lý AI chuyên cung cấp thông tin về Câu lạc bộ Lập trình ProPTIT.
    Bạn sẽ nhận được dữ liệu ngữ cảnh (context) từ một hệ thống Retrieval-Augmented Generation (RAG) chứa các thông tin chính xác về CLB.

    Nguyên tắc trả lời:
    1. Chỉ sử dụng thông tin từ context được cung cấp để trả lời. Context sẽ được cung cấp ở đầu mỗi query của người dùng. Câu hỏi của người dùng nằm ở cuối. 
    2. Nếu người dùng hỏi câu hỏi không liên quan đến CLB ProPTIT, hãy trả lời như bình thường, nhưng không sử dụng thông tin từ context
    3. Trình bày câu trả lời rõ ràng, dễ hiểu. Có thể sử dụng emoij icon khi cần.
    4. Tuyệt đối không suy đoán hoặc bịa thông tin.
    5. Giữ phong cách trả lời thân thiện, chuyên nghiệp và nhất quán.
    6. Trong context có thể chứa nhiều thông tin khác nhau, hãy tập trung vào câu hỏi của người dùng để trả lời chính xác nhất.

    Nhiệm vụ của bạn:
    - Trả lời các câu hỏi về CLB Lập trình ProPTIT: lịch sử, thành viên, hoạt động, sự kiện, dự án, nội quy, thành viên tiêu biểu, và các thông tin liên quan khác.
    """
                    },
                    {
                        "role": "user",
                        "content": context + "\n\nCâu hỏi: " + query
                    }
                ]
                
                # Gọi API để lấy câu trả lời
                response_obj = client.chat.completions.create(
                    model=llm_model,
                    messages=messages
                )
                response = response_obj.choices[0].message.content.strip()
                # print(response)
                _response_cache[cache_key] = response
                time.sleep(21)

            # Tách response thành các câu
            sentences = response.split('. ')
            print(sentences)
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Tạo prompt để đánh giá độ nhạy cảm
                messages_sensitivity = [
                    {
                        "role": "system",
                        "content": """Bạn là một chuyên gia đánh giá độ nhạy cảm của câu trả lời trong hệ thống RAG, có nhiệm vụ phân loại từng câu của câu trả lời dựa trên ngữ cảnh đã cho.
                    Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trong phần trả lời từ mô hình AI. Nhiệm vụ của bạn là đánh giá câu trả lời đó dựa trên ngữ cảnh và câu hỏi.
                    Input:
                    Question: Câu hỏi của người dùng
                    Contexts: Một hoặc nhiều đoạn văn bản được truy xuất
                    Answer: Chỉ một câu trong đoạn văn bản LLM sinh ra
                    Bạn hãy đánh giá dựa trên các nhãn sau: 
                    1: Nội dung câu được ngữ cảnh hỗ trợ hoặc suy ra trực tiếp.
                    0: Nội dung câu không được ngữ cảnh hỗ trợ, và không thể suy ra từ đó.
                    Ví dụ:
                    Question: Bạn có thể cho tôi biết về lịch sử của Câu lạc bộ Lập trình ProPTIT không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011.
                    1
                    Question: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011. Bạn có biết ngày cụ thể không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT là CLB thuộc PTIT.
                    0"""
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nContexts: {context}\n\nAnswer: {sentence.strip()}"
                    }
                ]
                
                # Gọi API để đánh giá độ nhạy cảm
                try:
                    sensitivity_response = client.chat.completions.create(
                        model=llm_model,
                        messages=messages_sensitivity
                    )
                    sensitivity_reply = sensitivity_response.choices[0].message.content.strip()
                    if sensitivity_reply == "0":
                        hits += 1
                except Exception as e:
                    print(f"Lỗi gọi API sensitivity cho câu {index}: {str(e)}")
                    continue
                time.sleep(30)
            total_sensitivity += hits / len(sentences) if len(sentences) > 0 else 0
                
        except Exception as e:
            print(f"Lỗi xử lý query {index}: {str(e)}")
            continue
    print(total_sensitivity / total_queries)    
    return total_sensitivity / total_queries


# Hàm để tính toán toàn bộ metrics trong module LLM Answer

def calculate_metrics_llm_answer(file_clb_proptit, file_train, embedding, vector_db, train):
    # Tạo ra 1 bảng csv, cột thứ nhất là K value, các cột còn lại là metrics. Sẽ có 3 hàng tương trưng với k = 3, 5, 7
    k_values = [3, 5, 7]
    metrics = {
        "K": [],
        "string_presence@k": [],
        "rouge_l@k": [],
        "bleu_4@k": [],
        "groundedness@k": [],
        "response_relevancy@k": [],
        "noise_sensitivity@k": []
    }
    # Lưu 2 chữ số thập phân cho các metrics
    for k in k_values:
        metrics["K"].append(k)
        metrics["string_presence@k"].append(round(string_presence_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["rouge_l@k"].append(round(rouge_l_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["bleu_4@k"].append(round(bleu_4_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["groundedness@k"].append(round(groundedness_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["response_relevancy@k"].append(round(response_relevancy_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["noise_sensitivity@k"].append(round(noise_sensitivity_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
    # Chuyển đổi metrics thành DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
    if train:
        metrics_df.to_csv("metrics_llm_answer_train.csv", index=False)
    else:
        metrics_df.to_csv("metrics_llm_answer_test.csv", index=False)
    return metrics_df

