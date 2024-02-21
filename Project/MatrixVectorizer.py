class MatrixVectorizer:
    def __init__(self):
        pass

    @staticmethod
    def vectorize(matrix, include_diagonal=False):
        # Determine the size of the matrix
        matrix_size = matrix.shape[0]

        # Initialize an empty list to store the vector elements
        vector_elements = []

        # Traverse the matrix column by column
        for col in range(matrix_size):
            for row in range(matrix_size):
                if row != col:  # Exclude diagonal if not including it
                    if row < col:
                        # Collect elements from the upper triangle
                        vector_elements.append(matrix[row, col])
                    elif include_diagonal and row == col + 1:
                        # Include the diagonal element immediately below the diagonal if specified
                        vector_elements.append(matrix[row, col])

        return np.array(vector_elements)

    @staticmethod
    def anti_vectorize(vector, matrix_size, include_diagonal=False):
        # Initialize the matrix with zeros
        matrix = np.zeros((matrix_size, matrix_size))

        # Counter for elements in the vector
        vector_idx = 0

        # Fill the matrix vertically, excluding the diagonal if specified
        for col in range(matrix_size):
            for row in range(matrix_size):
                if row != col:  # Skip diagonal if not including diagonal
                    if row < col:
                        # Fill in the upper triangle based on vector indexing
                        matrix[row, col] = vector[vector_idx]
                        # Reflect the value to the lower triangle
                        matrix[col, row] = vector[vector_idx]
                        vector_idx += 1
                    elif include_diagonal and row == col + 1:
                        # If including diagonal, fill it after completing each column
                        matrix[row, col] = vector[vector_idx]
                        matrix[col, row] = vector[vector_idx]
                        vector_idx += 1

        return matrix