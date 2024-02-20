def anti_vectorize(vector, matrix_size, include_diagonal=False):
    # Initialize the matrix with zeros
    matrix = np.zeros((matrix_size, matrix_size))
    
    # Fill the matrix
    if include_diagonal:
        # If the diagonal is included, just reshape the vector to a matrix
        indices = np.triu_indices_from(matrix)
        matrix[indices] = vector
        matrix += np.triu(matrix, 1).T  # Fill in the lower triangle
    else:
        # If the diagonal is not included, fill in the upper triangle only
        indices = np.triu_indices_from(matrix, k=1)
        matrix[indices] = vector
        matrix = matrix + matrix.T  # Fill in the lower triangle and diagonal

    return matrix
