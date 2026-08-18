# Failure Analysis Report

This report analyzes the bottom-N worst questions from the RAGAS evaluation to identify patterns and root causes of performance issues.

## Bottom 5 Worst Questions

1. **Question**: "Nhân viên được nghỉ phép bao nhiêu ngày?"
   - **Worst Metric**: context_recall
   - **Score**: 0.45
   - **Diagnosis**: Missing relevant chunks
   - **Suggested Fix**: Improve chunking strategy or add BM25 search to capture more relevant information

2. **Question**: "Mật khẩu thay đổi mỗi bao lâu?"
   - **Worst Metric**: answer_relevancy
   - **Score**: 0.52
   - **Diagnosis**: Answer doesn't match question
   - **Suggested Fix**: Improve prompt template to better align with question intent

3. **Question**: "Thời gian thử việc là bao lâu?"
   - **Worst Metric**: faithfulness
   - **Score**: 0.58
   - **Diagnosis**: LLM hallucinating
   - **Suggested Fix**: Tighten prompt and lower temperature for more factual responses

4. **Question**: "Nghỉ phép năm có tăng thêm không?"
   - **Worst Metric**: context_precision
   - **Score**: 0.61
   - **Diagnosis**: Too many irrelevant chunks
   - **Suggested Fix**: Add reranking or metadata filter to improve precision

5. **Question**: "Có cần giấy xác nhận y tế khi nghỉ ốm không?"
   - **Worst Metric**: context_recall
   - **Score**: 0.65
   - **Diagnosis**: Missing relevant chunks
   - **Suggested Fix**: Improve chunking or add BM25 search to capture more relevant information

## Summary

The main issues identified are:
- Context recall problems with key HR policies
- Answer relevancy issues with specific queries
- Faithfulness concerns with factual information
- Context precision challenges with irrelevant results

These issues can be addressed through improved chunking strategies, better reranking, and refined prompting techniques.