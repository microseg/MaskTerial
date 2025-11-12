/**
 * API Client Utility
 * Automatically adds Authorization header (JWT token) to all API requests
 */

const API_BASE_URL = '';

/**
 * Get current user ID from localStorage
 */
export function getCurrentUserId() {
  return localStorage.getItem('userId') || 'test_user';
}

/**
 * Get access token from localStorage
 */
export function getAccessToken() {
  return localStorage.getItem('accessToken');
}

/**
 * Get auth headers with Bearer token or fallback to X-User-ID
 */
export function getAuthHeaders() {
  const token = getAccessToken();
  const headers = {};
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  } else {
    // Fallback to X-User-ID header for backward compatibility
    headers['X-User-ID'] = getCurrentUserId();
  }
  
  return headers;
}

/**
 * Create FormData with user_id automatically included
 */
export function createFormDataWithUser(additionalData = {}) {
  const formData = new FormData();
  formData.append('user_id', getCurrentUserId());
  
  // Add all additional data
  Object.entries(additionalData).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      formData.append(key, value);
    }
  });
  
  return formData;
}

/**
 * Create URL with query parameters (no user_id in URL anymore)
 */
export function createUrlWithParams(endpoint, additionalParams = {}) {
  const url = new URL(endpoint, window.location.origin);
  
  Object.entries(additionalParams).forEach(([key, value]) => {
    if (value !== null && value !== undefined) {
      url.searchParams.append(key, value);
    }
  });
  
  return url.toString();
}

// Backward compatibility alias
export const createUrlWithUser = createUrlWithParams;

/**
 * Fetch wrapper that automatically adds auth headers
 */
export async function fetchWithAuth(url, options = {}) {
  const authHeaders = getAuthHeaders();
  
  const mergedOptions = {
    ...options,
    headers: {
      ...authHeaders,
      ...options.headers,
    },
  };
  
  return fetch(url, mergedOptions);
}

// Backward compatibility alias
export const fetchWithUser = fetchWithAuth;

/**
 * POST request with automatic auth headers
 */
export async function postWithUser(url, formData) {
  // If formData doesn't have user_id, add it (for backward compatibility)
  if (formData instanceof FormData && !formData.has('user_id')) {
    formData.append('user_id', getCurrentUserId());
  }
  
  const response = await fetch(url, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData,
  });
  
  return response;
}

/**
 * GET request with automatic auth headers
 */
export async function getWithUser(endpoint, additionalParams = {}) {
  const url = createUrlWithParams(endpoint, additionalParams);
  
  const response = await fetch(url, {
    method: 'GET',
    headers: getAuthHeaders(),
  });
  
  return response;
}

/**
 * Upload image to S3 and store metadata in DynamoDB
 * @param {File} imageFile - The image file to upload
 * @param {string} imageName - Custom name for the image (optional)
 * @returns {Promise<Object>} Upload result with imageID and S3 details
 */
export async function uploadImage(imageFile, imageName = null) {
  const formData = new FormData();
  formData.append('image_file', imageFile);
  
  if (imageName) {
    formData.append('image_name', imageName);
  }
  
  formData.append('user_id', getCurrentUserId());
  
  const response = await fetch('/api/upload_image', {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to upload image');
  }
  
  return response.json();
}

/**
 * Get list of user's uploaded images with pagination support
 * @param {Object} options - Query options
 * @param {number} options.limit - Number of items per page (default: 5)
 * @param {string|null} options.lastKey - Pagination token from previous request
 * @param {string} options.status - Filter by status ('active' or 'deleted') - if using old API
 * @returns {Promise<{items: Array, last_evaluated_key: string|null, count: number, has_more: boolean}>}
 */
export async function getUserImages({ limit = 5, lastKey = null, status = 'active' } = {}) {
  const params = { limit };
  
  // Add pagination token if provided
  if (lastKey) {
    params.last_key = lastKey;
  }
  
  // Add status if needed (for backward compatibility)
  if (status) {
    params.status = status;
  }
  
  const url = createUrlWithParams('/api/images', params);
  
  const response = await fetch(url, {
    method: 'GET',
    headers: getAuthHeaders(),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to retrieve images');
  }
  
  return response.json();
}

/**
 * Get details of a specific image
 * @param {string} imageId - Image identifier
 * @returns {Promise<Object>} Image details
 */
export async function getImageById(imageId) {
  const url = `/api/images/${imageId}`;
  
  const response = await fetch(url, {
    method: 'GET',
    headers: getAuthHeaders(),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch image');
  }
  
  return response.json();
}

/**
 * Save image metadata to DynamoDB
 * @param {Object} imageData - Image data
 * @param {string} imageData.image_name - Name of the image
 * @param {string} [imageData.image_id] - Optional image ID
 * @param {string} [imageData.image_url] - Optional image URL
 * @param {Object} [imageData.metadata] - Optional additional metadata
 * @returns {Promise<{success: boolean, image_id: string}>}
 */
export async function saveImageMetadata(imageData) {
  const formData = createFormDataWithUser({
    image_name: imageData.image_name,
    image_id: imageData.image_id,
    image_url: imageData.image_url,
  });
  
  // Add metadata as JSON string if provided
  if (imageData.metadata) {
    formData.append('metadata', JSON.stringify(imageData.metadata));
  }
  
  const response = await postWithUser('/api/images', formData);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to save image');
  }
  
  return response.json();
}

/**
 * Delete an image from DynamoDB
 * @param {string} imageId - Image identifier
 * @returns {Promise<{success: boolean}>}
 */
export async function deleteImageById(imageId) {
  const url = `/api/images/${imageId}`;
  
  const response = await fetch(url, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to delete image');
  }
  
  return response.json();
}

/**
 * Update image metadata in DynamoDB
 * @param {string} imageId - Image identifier
 * @param {Object} updateData - Data to update
 * @param {string} [updateData.image_name] - New image name
 * @param {string} [updateData.status] - New status (active | deleted)
 * @param {string} [updateData.type] - Image type (UPLOADED | PROCESSED)
 * @param {Object} [updateData.metadata] - Metadata object to update
 * @returns {Promise<{success: boolean, updated_fields: Array}>}
 */
export async function updateImageMetadata(imageId, updateData) {
  const formData = createFormDataWithUser();
  
  if (updateData.image_name) {
    formData.append('image_name', updateData.image_name);
  }
  
  if (updateData.status) {
    // Validate status
    if (!['active', 'deleted'].includes(updateData.status)) {
      throw new Error('Status must be either "active" or "deleted"');
    }
    formData.append('status', updateData.status);
  }
  
  if (updateData.type) {
    // Validate type
    if (!['UPLOADED', 'PROCESSED'].includes(updateData.type)) {
      throw new Error('Type must be either "UPLOADED" or "PROCESSED"');
    }
    formData.append('type', updateData.type);
  }
  
  if (updateData.metadata) {
    formData.append('metadata', JSON.stringify(updateData.metadata));
  }
  
  const url = `/api/images/${imageId}`;
  
  const response = await fetch(url, {
    method: 'PUT',
    headers: getAuthHeaders(),
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to update image');
  }
  
  return response.json();
}

/**
 * Refresh the presigned download URL for an image
 * @param {string} imageId - Image identifier
 * @param {number} expirationDays - URL expiration in days (default: 7)
 * @returns {Promise<{downloadURL: string, downloadURLExpiresAt: number}>}
 */
export async function refreshDownloadUrl(imageId, expirationDays = 7) {
  const formData = createFormDataWithUser({
    expiration: expirationDays
  });
  
  const url = `/api/images/${imageId}/refresh-download-url`;
  
  const response = await fetch(url, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to refresh download URL');
  }
  
  return response.json();
}

export default {
  getCurrentUserId,
  createFormDataWithUser,
  createUrlWithUser,
  fetchWithUser,
  postWithUser,
  getWithUser,
  uploadImage,
  getUserImages,
  getImageById,
  saveImageMetadata,
  updateImageMetadata,
  deleteImageById,
  refreshDownloadUrl,
};

