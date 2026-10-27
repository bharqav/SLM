# Training Data Documentation

## Dataset Overview

The from-scratch SLM was trained on a curated dataset of public code prose and technical documentation.

### Data Sources

1. **GitHub Code Repositories**
   - Filtered for permissively licensed repositories (MIT, Apache 2.0, BSD)
   - Excluded large binary files and generated code
   - Focus on documentation, comments, and prose in code

2. **Technical Documentation**
   - Open-source project READMEs
   - API documentation
   - Tutorials and guides

3. **Code Documentation**
   - Docstrings and comments
   - Technical blogs and articles
   - Forum discussions (filtered for quality)

### Data Processing Pipeline

1. **Collection**: GitHub Archive data, filtered by language and license
2. **Deduplication**: Remove near-duplicate content
3. **Filtering**: 
   - Minimum content length (100 characters)
   - Maximum content length (8192 tokens)
   - Remove boilerplate, navigation, ads
   - Keep only substantive technical content

4. **Tokenization**: 
   - Using tiktoken GPT-2 tokenizer
   - Token count: ~13.2M total tokens

5. **Splitting**:
   - Training: 12.5M tokens (95%)
   - Validation: 700K tokens (5%)

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total raw tokens (estimated) | 13.2M |
| Total processed tokens | 13.2M |
| Training tokens | 12.5M |
| Validation tokens | 700K |
| Training chunks | 1 |
| Validation chunks | 1 |
| Average document length | ~2K tokens |

### Sample Data

#### Example 1: Python Function Docstring
```python
def calculate_factorial(n):
    """Calculate the factorial of a non-negative integer n.
    
    Args:
        n (int): Non-negative integer
        
    Returns:
        int: Factorial of n, or None if n is negative
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

#### Example 2: API Documentation
```
Endpoint: GET /users/{id}
Description: Retrieve user information by ID
Parameters:
  - id (required): User ID (integer)
Responses:
  200: User object with name, email, and metadata
  404: User not found
```

#### Example 3: Technical Explanation
```
Attention Mechanism in Transformers

The attention mechanism allows a model to focus on different parts
of the input when producing each output element. For each position,
the model computes a set of attention weights that determine how
much focus to place on each other position in the input.

The scaled dot-product attention is computed as:
  attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

where Q, K, V are query, key, and value matrices, and d_k
is the dimension of the key vectors.
```