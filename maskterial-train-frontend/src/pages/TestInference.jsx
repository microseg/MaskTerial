import styles from "./TestInference.module.css";
import { useState, useEffect } from "react";
import { ImageDropzone } from "../components/ImageDropzone";
import { CanvasImage } from "../components/CanvasImage";
import { Paper, Select, Button, ActionIcon, Text, ScrollArea, Tooltip, Badge, Group, Modal, TextInput, Stack } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconChevronLeft, IconChevronRight, IconPhoto, IconRefresh, IconDownload, IconEdit } from "@tabler/icons-react";
import { getUserImages, saveImageMetadata, deleteImageById, refreshDownloadUrl, updateImageMetadata } from "../utils/apiClient";

const formatModelData = (data) => {
  return Object.keys(data).reduce((acc, model) => {
    const materials = data[model];
    return [...acc, ...materials.map((material) => `${model}-${material}`)];
  }, []);
};

export function TestInference() {
  const [isLoading, setIsLoading] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);
  const [currentImageURL, setCurrentImageURL] = useState(null);
  const [uploadedImageData, setUploadedImageData] = useState(null);
  const [availableSegModels, setAvailableSegModels] = useState([]);
  const [availableClsModels, setAvailableClsModels] = useState([]);
  const [availablePPModels, setAvailablePPModels] = useState([]);
  const [selectedSegModel, setSelectedSegModel] = useState(null);
  const [selectedClsModel, setSelectedClsModel] = useState(null);
  const [selectedPPModel, setSelectedPPModel] = useState(null);
  const [inferenceResults, setInferenceResults] = useState([]);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);
  const [uploadedImages, setUploadedImages] = useState([]);
  const [isLoadingGallery, setIsLoadingGallery] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageTokens, setPageTokens] = useState([null]); // [null, token1, token2, ...]
  const [hasNextPage, setHasNextPage] = useState(false);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [editingImage, setEditingImage] = useState(null);
  const [editImageName, setEditImageName] = useState('');

  const handleUserImageInput = async (files, uploadResult) => {
    // Use local blob URL for immediate display
    const localURL = URL.createObjectURL(files[0]);
    
    setCurrentImageURL(localURL);
    setCurrentImage(files[0]);
    
    // Store uploaded image data if available
    if (uploadResult) {
      setUploadedImageData(uploadResult);
      console.log("Image uploaded to S3:", uploadResult);
      
      // Refresh gallery to show the new image (jump to page 1)
      // New uploads appear at the top (most recent first)
      setTimeout(() => {
        handleRefreshGallery();
        notifications.show({
          title: "Gallery Updated",
          message: "Jumped to page 1 to show your new upload",
          color: "blue",
          autoClose: 2000,
        });
      }, 500); // Small delay to ensure upload is fully completed
      
      console.log("Gallery will refresh to show new upload");
    } else {
      // No upload result, use local blob URL (already created above)
      const newImage = {
        id: Date.now(),
        file: files[0],
        url: localURL,
        name: files[0].name,
        uploadResult: null
      };
      setUploadedImages((prev) => [newImage, ...prev]);
      console.log("Image added to gallery with local URL");
    }
  };

  const handleUploadSuccess = (result, file) => {
    console.log("Upload successful:", result);
    notifications.show({
      title: "Upload Success",
      message: `${file.name} uploaded successfully`,
      color: "green",
      autoClose: 3000,
    });
  };

  const handleUploadError = (error, file) => {
    console.error("Upload failed:", error);
    // Can still display the image locally even if upload fails
    setCurrentImageURL(URL.createObjectURL(file));
    setCurrentImage(file);
  };

  const runInference = () => {
    if (isLoading) {
      return;
    }

    if (!currentImage) {
      return;
    }

    if (!selectedSegModel && !selectedClsModel) {
      return;
    }

    setIsLoading(true);

    // send a POST request to the server with the selected models and the image
    let formData = new FormData();
    formData.append("files", currentImage, currentImage.name);
    if (selectedSegModel) {
      formData.append("segmentation_model", selectedSegModel);
    }
    if (selectedClsModel) {
      formData.append("classification_model", selectedClsModel);
    }
    if (selectedPPModel) {
      formData.append("postprocessing_model", selectedPPModel);
    }
    formData.append("score_threshold", 0.3);
    formData.append("return_bbox", true);

    fetch(import.meta.env.VITE_INFERENCE_URL, {
      method: "POST",
      body: formData,
    })
      .then(async (response) => {
        const status = response.status;
        const data = await response.json();
        return { status, data };
      })
      .then(async ({ status, data }) => {
        console.log(data);
        if (status === 200) {
          notifications.show({
            title: "Success",
            message: "Results received successfully",
            color: "blue",
            autoClose: false,
          });
          setInferenceResults(data);
          
          // Update DynamoDB if image has imageID (from S3 upload)
          if (uploadedImageData?.imageID) {
            try {
              await updateImageMetadata(uploadedImageData.imageID, {
                type: 'PROCESSED',
                metadata: {
                  inference_results: data,
                  inference_at: new Date().toISOString(),
                  segmentation_model: selectedSegModel,
                  classification_model: selectedClsModel,
                  postprocessing_model: selectedPPModel,
                  flake_count: data.length,
                  materials_detected: data.map(item => item.material_class).filter(Boolean)
                }
              });
              
              console.log('Image metadata updated with inference results');
              
              // Update local gallery state to show PROCESSED status immediately
              setUploadedImages(prev => prev.map(img => 
                img.imageID === uploadedImageData.imageID
                  ? {
                      ...img,
                      metadata: {
                        ...img.metadata,
                        type: 'PROCESSED',
                        flake_count: data.length,
                        inference_at: new Date().toISOString(),
                        segmentation_model: selectedSegModel,
                        classification_model: selectedClsModel,
                        postprocessing_model: selectedPPModel,
                        inference_results: data
                      }
                    }
                  : img
              ));
              
              console.log('Gallery updated locally to show PROCESSED status');
              
            } catch (error) {
              console.error('Failed to update image metadata:', error);
              // Don't show error to user, inference was successful
            }
          }
        } else {
          notifications.show({
            title: "Error",
            message: `Hmm, something went wrong: ${data.detail}`,
            color: "red",
            autoClose: false,
          });
        }
      })
      .finally(() => {
        setIsLoading(false);
      });
  };

  // Load gallery images from DynamoDB with pagination
  const loadGalleryPage = async (pageNum) => {
    if (isLoadingGallery) return;
    
    setIsLoadingGallery(true);
    try {
      // Get token for this page (pageNum - 1 because array is 0-indexed)
      const pageToken = pageTokens[pageNum - 1];
      
      const result = await getUserImages({
        limit: 5,  // 每页5张（测试用）
        lastKey: pageToken
      });
      
      // Convert DynamoDB items to gallery format
      const dbImages = result.items.map(item => {
        // Use backend proxy URL to avoid CORS issues
        const proxyUrl = `/api/images/${item.imageID}/download`;
        
        return {
          id: item.imageID,
          imageID: item.imageID,
          url: proxyUrl,  // Use backend proxy URL (no CORS issues)
          name: item.image_name,
          createdAt: item.CreatedAt,
          metadata: {
            ...item.metadata,
            type: item.type  // Include type from DynamoDB
          },
          download_url: proxyUrl,  // Use same proxy URL for download
          s3Key: item.s3Key,
          s3Url: item.image_url,  // Keep original S3 URL for reference
          uploadResult: {
            imageID: item.imageID,
            imageURL: proxyUrl,
            downloadURL: proxyUrl,
            type: item.type,  // Include type
            bucket: item.metadata?.s3_bucket,
            key: item.s3Key || item.metadata?.s3_key
          }
        };
      });
      
      // Replace current page images (not append)
      setUploadedImages(dbImages);
      
      // Update hasNextPage - 只检查nextPageToken和hasMore
      // 不能只依赖当前页有没有数据，因为可能总数刚好是5的倍数
      const hasNext = (
        (result.nextPageToken !== null && result.nextPageToken !== undefined) || 
        (result.hasMore === true)
      );
      
      setHasNextPage(hasNext);
      
      // 更新分页token - 关键修复
      if (result.nextPageToken) {
        // 如果有nextPageToken，更新或添加到tokens数组
        setPageTokens(prev => {
          const newTokens = [...prev];
          // 确保tokens数组足够长
          while (newTokens.length < pageNum) {
            newTokens.push(null);
          }
          // 设置下一页的token
          if (newTokens.length === pageNum) {
            newTokens.push(result.nextPageToken);
          } else {
            newTokens[pageNum] = result.nextPageToken;
          }
          console.log(`Updated page tokens for page ${pageNum + 1}:`, newTokens);
          return newTokens;
        });
      } else {
        // 没有nextPageToken，确保没有多余的token
        setPageTokens(prev => {
          if (prev.length > pageNum) {
            return prev.slice(0, pageNum);
          }
          return prev;
        });
      }
      
      setCurrentPage(pageNum);
      
      console.log(`Loaded page ${pageNum}: ${dbImages.length} images, hasNext: ${hasNext}, nextPageToken: ${result.nextPageToken ? 'exists' : 'null'}`);
      
      // 如果加载的页面是空的，处理空页面情况
      if (dbImages.length === 0) {
        // 空页面意味着没有更多数据
        setHasNextPage(false);
        
        if (pageNum > 1) {
          // 不是第1页，自动跳回前一页
          console.warn(`Page ${pageNum} is empty, jumping back to page ${pageNum - 1}`);
          // 清理这个空页面的token
          setPageTokens(prev => prev.slice(0, pageNum - 1));
          setTimeout(() => {
            loadGalleryPage(pageNum - 1);
          }, 100);
        } else {
          // 第1页也是空的，说明没有任何数据
          console.log('No images found in gallery');
        }
      }
      
    } catch (error) {
      console.warn("DynamoDB gallery not available, using local mode:", error);
      // Gracefully degrade to local-only mode
      if (!error.message.includes('Not Found') && !error.message.includes('404')) {
        notifications.show({
          title: "Info",
          message: "Using local gallery mode",
          color: "blue",
          autoClose: 3000,
        });
      }
      setHasNextPage(false);
    } finally {
      setIsLoadingGallery(false);
    }
  };

  useEffect(() => {
    // Load available models
    fetch(import.meta.env.VITE_AVAILABLE_MODELS_URL)
      .then((response) => response.json())
      .then((data) => {
        setAvailableSegModels(
          formatModelData(data.available_models.segmentation_models)
        );
        setAvailableClsModels(
          formatModelData(data.available_models.classification_models)
        );
        setAvailablePPModels(
          formatModelData(data.available_models.postprocessing_models)
        );
      });
    
    // Load first page of gallery images from DynamoDB
    loadGalleryPage(1);
  }, []);

  const ImageSection = (
    <>
      {currentImageURL && (
        <CanvasImage src={currentImageURL} flakes={inferenceResults} />
      )}
    </>
  );

  const controlSection = (
    <Paper p="md" shadow="xs" withBorder className={styles.controlPaper}>
      <Select
        data={availableSegModels}
        label="Segmentation Model"
        placeholder="None"
        value={selectedSegModel}
        onChange={setSelectedSegModel}
        clearable
      />
      <Select
        data={availableClsModels}
        label="Classification Model"
        placeholder="None"
        value={selectedClsModel}
        onChange={setSelectedClsModel}
        clearable
      />
      <Select
        data={availablePPModels}
        label="Postprocessing Model"
        placeholder="None"
        value={selectedPPModel}
        onChange={setSelectedPPModel}
        clearable
      />
      <Button
        color="blue"
        className={styles.inferenceButton}
        onClick={runInference}
        disabled={
          !currentImage || (!selectedSegModel && !selectedClsModel) || isLoading
        }
        loading={isLoading}
      >
        Run Inference
      </Button>
    </Paper>
  );

  const dropzoneSection = (
    <Paper p="md" shadow="xs" withBorder className={styles.dropzonePaper}>
      <ImageDropzone
        handleImageUpload={handleUserImageInput}
        autoUpload={true}
        onUploadSuccess={handleUploadSuccess}
        onUploadError={handleUploadError}
        showNotifications={true}
        className={styles.dropzone}
      />
    </Paper>
  );

  const handleSelectImage = async (image) => {
    // 先清除旧状态，确保Canvas完全重新渲染
    setInferenceResults([]);
    setCurrentImageURL(null);
    setCurrentImage(null);
    
    // 短暂延迟后设置新图片，确保Canvas已清空
    setTimeout(() => {
      setCurrentImageURL(image.url);
    }, 50);
    
    // Check if image is PROCESSED and has inference results
    const imageType = image.metadata?.type || image.uploadResult?.type;
    const hasInferenceResults = image.metadata?.inference_results && 
                                Array.isArray(image.metadata.inference_results) && 
                                image.metadata.inference_results.length > 0;
    
    // If image is PROCESSED, load inference results from metadata
    if (imageType === 'PROCESSED' && hasInferenceResults) {
      notifications.show({
        title: "Loading Results",
        message: "Displaying saved inference results...",
        color: "blue",
        autoClose: 2000,
      });
      
      // Set inference results from metadata (稍后设置，让图片先加载)
      setTimeout(() => {
        setInferenceResults(image.metadata.inference_results);
      }, 100);
      
      console.log('Loaded inference results from DynamoDB:', {
        flake_count: image.metadata.flake_count,
        inference_at: image.metadata.inference_at,
        models: {
          segmentation: image.metadata.segmentation_model,
          classification: image.metadata.classification_model
        }
      });
      
      // Also display the models that were used
      notifications.show({
        title: "Inference Results Loaded",
        message: `Found ${image.metadata.flake_count || 0} flakes (${image.metadata.inference_at ? new Date(image.metadata.inference_at).toLocaleString() : 'Unknown time'})`,
        color: "green",
        autoClose: 5000,
      });
    }
    
    // If image has a file object (locally uploaded), use it directly
    if (image.file) {
      setTimeout(() => {
        setCurrentImage(image.file);
        setUploadedImageData(image.uploadResult);
      }, 50);
    } else if (image.url) {
      // If no file but has URL (from DynamoDB), download it
      try {
        // Fetch the image from URL
        const response = await fetch(image.url);
        if (!response.ok) {
          throw new Error('Failed to fetch image');
        }
        
        const blob = await response.blob();
        
        // Create a File object from the blob
        const file = new File([blob], image.name || 'image.jpg', {
          type: blob.type || 'image/jpeg'
        });
        
        setTimeout(() => {
          setCurrentImage(file);
          setUploadedImageData(image.uploadResult);
        }, 50);
        
        console.log('Image downloaded from URL and ready for inference');
        
      } catch (error) {
        console.error('Failed to download image from URL:', error);
        notifications.show({
          title: "Error",
          message: "Failed to load image from server",
          color: "red",
          autoClose: 3000,
        });
      }
    }
  };

  const handleDeleteImage = async (imageId) => {
    try {
      // If image has imageID (from DynamoDB), delete from DB
      const image = uploadedImages.find(img => img.id === imageId);
      if (image && image.imageID) {
        await deleteImageById(image.imageID);
        notifications.show({
          title: "Success",
          message: "Image deleted successfully",
          color: "green",
          autoClose: 3000,
        });
      }
      
      // Remove from local state
      const newImages = uploadedImages.filter((img) => img.id !== imageId);
      setUploadedImages(newImages);
      
      // If deleted image is currently displayed, clear it
      if (uploadedImages.find((img) => img.id === imageId && img.url === currentImageURL)) {
        setCurrentImageURL(null);
        setCurrentImage(null);
        setInferenceResults([]);
      }
      
      // 智能分页处理
      if (image && image.imageID) {
        // 如果当前页删除后为空
        if (newImages.length === 0) {
          if (currentPage > 1) {
            // 不是第1页，跳转到前一页
            loadGalleryPage(currentPage - 1);
            console.log(`Page ${currentPage} empty, jumped to page ${currentPage - 1}`);
          } else {
            // 是第1页，刷新以查看是否还有数据
            loadGalleryPage(1);
            console.log('Page 1 refreshed after deletion');
          }
        } else {
          // 当前页还有图片，刷新当前页以补充新数据
          loadGalleryPage(currentPage);
          console.log(`Page ${currentPage} refreshed to load more items`);
        }
      }
      
    } catch (error) {
      console.error("Failed to delete image:", error);
      notifications.show({
        title: "Error",
        message: "Failed to delete image",
        color: "red",
        autoClose: 3000,
      });
    }
  };
  
  const handleRefreshGallery = () => {
    // Reset to page 1
    setCurrentPage(1);
    setPageTokens([null]);
    setHasNextPage(false);
    loadGalleryPage(1);
  };
  
  const handleNextPage = () => {
    if (hasNextPage && !isLoadingGallery) {
      loadGalleryPage(currentPage + 1);
    }
  };
  
  const handlePrevPage = () => {
    if (currentPage > 1 && !isLoadingGallery) {
      loadGalleryPage(currentPage - 1);
    }
  };
  
  const handleOpenEditModal = (image, e) => {
    e.stopPropagation();
    setEditingImage(image);
    setEditImageName(image.name || '');
    setEditModalOpen(true);
  };
  
  const handleCloseEditModal = () => {
    setEditModalOpen(false);
    setEditingImage(null);
    setEditImageName('');
  };
  
  const handleSaveEdit = async () => {
    if (!editingImage?.imageID) {
      notifications.show({
        title: "Error",
        message: "Cannot update local-only image",
        color: "red",
        autoClose: 3000,
      });
      return;
    }
    
    if (!editImageName.trim()) {
      notifications.show({
        title: "Error",
        message: "Image name cannot be empty",
        color: "red",
        autoClose: 3000,
      });
      return;
    }
    
    try {
      await updateImageMetadata(editingImage.imageID, {
        image_name: editImageName
      });
      
      notifications.show({
        title: "Success",
        message: "Image name updated successfully",
        color: "green",
        autoClose: 3000,
      });
      
      // Update local state
      setUploadedImages(prev => prev.map(img => 
        img.id === editingImage.id 
          ? { ...img, name: editImageName }
          : img
      ));
      
      // Close modal
      handleCloseEditModal();
      
    } catch (error) {
      console.error("Failed to update image:", error);
      notifications.show({
        title: "Error",
        message: error.message || "Failed to update image",
        color: "red",
        autoClose: 3000,
      });
    }
  };
  
  const handleDownloadImage = async (image, e) => {
    e.stopPropagation();
    
    try {
      // Use download URL (permanent, no expiration check needed)
      const downloadUrl = image.download_url || image.url;
      
      if (downloadUrl) {
        // Open URL in new tab to download
        window.open(downloadUrl, '_blank');
        
        notifications.show({
          title: "Download Started",
          message: "Image download started",
          color: "green",
          autoClose: 2000,
        });
      } else {
        notifications.show({
          title: "No URL",
          message: "Download URL not available",
          color: "orange",
          autoClose: 3000,
        });
      }
      
    } catch (error) {
      console.error("Failed to download image:", error);
      notifications.show({
        title: "Download Failed",
        message: error.message || "Failed to download image",
        color: "red",
        autoClose: 3000,
      });
    }
  };
  

  const gallerySection = (
    <div className={`${styles.gallery} ${isGalleryOpen ? styles.galleryOpen : styles.galleryClosed}`}>
      <div className={styles.galleryHeader}>
        <div className={styles.galleryTitle}>
          <IconPhoto size={20} />
          <Text size="sm" fw={600}>Image Gallery</Text>
          <ActionIcon
            variant="subtle"
            size="sm"
            onClick={handleRefreshGallery}
            loading={isLoadingGallery}
            title="Refresh gallery"
          >
            <IconRefresh size={16} />
          </ActionIcon>
        </div>
        <ActionIcon
          variant="subtle"
          onClick={() => setIsGalleryOpen(false)}
          className={styles.closeButton}
        >
          <IconChevronLeft size={18} />
        </ActionIcon>
      </div>
      <ScrollArea className={styles.galleryContent}>
        {uploadedImages.length === 0 && !isLoadingGallery ? (
          <Text size="sm" c="dimmed" ta="center" mt="md">
            No images uploaded yet
          </Text>
        ) : (
          <>
            <div className={styles.imageGrid}>
              {uploadedImages.map((image) => (
                <div
                  key={image.id}
                  className={`${styles.galleryImageItem} ${
                    currentImageURL === image.url ? styles.selectedImage : ""
                  }`}
                  onClick={() => handleSelectImage(image)}
                >
                  <img src={image.url} alt={image.name} className={styles.galleryImage} />
                  <div className={styles.imageInfo}>
                    <Text size="xs" truncate title={image.name}>
                      {image.name}
                    </Text>
                    <Group gap="xs" mt={4}>
                      {/* Type badge */}
                      <Badge 
                        size="xs" 
                        variant="dot"
                        color={image.metadata?.type === 'PROCESSED' || image.uploadResult?.type === 'PROCESSED' ? 'green' : 'gray'}
                      >
                        {image.metadata?.type || image.uploadResult?.type || 'UPLOADED'}
                      </Badge>
                      
                      {/* Flake count if processed */}
                      {(image.metadata?.type === 'PROCESSED' || image.uploadResult?.type === 'PROCESSED') && 
                       image.metadata?.flake_count && (
                        <Text size="xs" c="dimmed">
                          {image.metadata.flake_count} flakes
                        </Text>
                      )}
                    </Group>
                  </div>
                  
                  {/* Action buttons */}
                  <div className={styles.imageActions}>
                    {/* Edit button */}
                    {image.imageID ? (
                      <Tooltip label="Edit Info">
                        <ActionIcon
                          size="xs"
                          color="violet"
                          variant="filled"
                          onClick={(e) => handleOpenEditModal(image, e)}
                        >
                          <IconEdit size={12} />
                        </ActionIcon>
                      </Tooltip>
                    ) : null}
                    
                    {/* Download button */}
                    {image.download_url || image.imageID ? (
                      <Tooltip label="Download">
                        <ActionIcon
                          size="xs"
                          color="blue"
                          variant="filled"
                          onClick={(e) => handleDownloadImage(image, e)}
                        >
                          <IconDownload size={12} />
                        </ActionIcon>
                      </Tooltip>
                    ) : null}
                    
                    {/* Delete button */}
                    <Tooltip label="Delete">
                      <ActionIcon
                        size="xs"
                        color="red"
                        variant="filled"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteImage(image.id);
                        }}
                      >
                        ×
                      </ActionIcon>
                    </Tooltip>
                  </div>
                </div>
              ))}
            </div>
            
            {/* Pagination Controls */}
            {(currentPage > 1 || hasNextPage) && (
              <Group justify="space-between" mt="sm" px="xs">
                <Button
                  variant="subtle"
                  size="xs"
                  onClick={handlePrevPage}
                  disabled={currentPage === 1 || isLoadingGallery}
                >
                  ← Prev
                </Button>
                
                <Text size="xs" fw={500} c="dimmed">
                  Page {currentPage} ({uploadedImages.length} items)
                </Text>
                
                <Button
                  variant="subtle"
                  size="xs"
                  onClick={handleNextPage}
                  disabled={!hasNextPage || isLoadingGallery}
                >
                  Next →
                </Button>
              </Group>
            )}
          </>
        )}
        {isLoadingGallery && uploadedImages.length === 0 && (
          <Text size="sm" c="dimmed" ta="center" mt="md">
            Loading images...
          </Text>
        )}
      </ScrollArea>
    </div>
  );

  const toggleButton = !isGalleryOpen && (
    <Paper className={styles.galleryToggleButton} shadow="sm" withBorder>
      <Button
        variant="subtle"
        color="blue"
        leftSection={<IconPhoto size={18} />}
        onClick={() => setIsGalleryOpen(true)}
        size="sm"
      >
        Gallery
      </Button>
    </Paper>
  );

  return (
    <>
      <div className={styles.gridContainer}>
        <div className={styles.leftSection}>
          {toggleButton}
          {gallerySection}
          <div className={styles.imageSection}>
            {currentImageURL ? ImageSection : <div className={styles.emptyImagePlaceholder}>Upload an image to start</div>}
          </div>
        </div>
        <div className={styles.controlSection}>
          {controlSection}
          {dropzoneSection}
        </div>
      </div>

      {/* Edit Image Modal */}
      <Modal
        opened={editModalOpen}
        onClose={handleCloseEditModal}
        title="Edit Image Information"
        size="md"
      >
        {editingImage && (
          <Stack gap="md">
            {/* Image Preview */}
            <div style={{ textAlign: 'center' }}>
              <img 
                src={editingImage.url} 
                alt={editingImage.name}
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '200px', 
                  borderRadius: '8px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}
              />
            </div>
            
            {/* Image Name Input */}
            <TextInput
              label="Image Name"
              placeholder="Enter image name"
              value={editImageName}
              onChange={(e) => setEditImageName(e.target.value)}
              required
            />
            
            {/* Display Image Info */}
            <div>
              <Text size="sm" fw={500} mb={8}>Image Information</Text>
              <Stack gap="xs">
                <Group justify="space-between">
                  <Text size="xs" c="dimmed">Image ID:</Text>
                  <Text size="xs" fw={500}>{editingImage.imageID}</Text>
                </Group>
                
                <Group justify="space-between">
                  <Text size="xs" c="dimmed">Type:</Text>
                  <Badge 
                    size="xs" 
                    color={editingImage.metadata?.type === 'PROCESSED' ? 'green' : 'gray'}
                  >
                    {editingImage.metadata?.type || 'UPLOADED'}
                  </Badge>
                </Group>
                
                <Group justify="space-between">
                  <Text size="xs" c="dimmed">Status:</Text>
                  <Badge size="xs" color="blue">
                    {editingImage.metadata?.status || 'active'}
                  </Badge>
                </Group>
                
                {editingImage.createdAt && (
                  <Group justify="space-between">
                    <Text size="xs" c="dimmed">Created:</Text>
                    <Text size="xs" fw={500}>
                      {new Date(editingImage.createdAt).toLocaleString()}
                    </Text>
                  </Group>
                )}
                
                {editingImage.metadata?.flake_count && (
                  <Group justify="space-between">
                    <Text size="xs" c="dimmed">Flakes Detected:</Text>
                    <Text size="xs" fw={500} c="green">
                      {editingImage.metadata.flake_count}
                    </Text>
                  </Group>
                )}
                
                {editingImage.metadata?.inference_at && (
                  <Group justify="space-between">
                    <Text size="xs" c="dimmed">Last Inference:</Text>
                    <Text size="xs" fw={500}>
                      {new Date(editingImage.metadata.inference_at).toLocaleString()}
                    </Text>
                  </Group>
                )}
                
                {editingImage.metadata?.segmentation_model && (
                  <Group justify="space-between">
                    <Text size="xs" c="dimmed">Seg Model:</Text>
                    <Text size="xs" fw={500}>
                      {editingImage.metadata.segmentation_model}
                    </Text>
                  </Group>
                )}
                
                {editingImage.metadata?.classification_model && (
                  <Group justify="space-between">
                    <Text size="xs" c="dimmed">Cls Model:</Text>
                    <Text size="xs" fw={500}>
                      {editingImage.metadata.classification_model}
                    </Text>
                  </Group>
                )}
              </Stack>
            </div>
            
            {/* Action Buttons */}
            <Group justify="flex-end" mt="md">
              <Button variant="subtle" onClick={handleCloseEditModal}>
                Cancel
              </Button>
              <Button onClick={handleSaveEdit}>
                Save Changes
              </Button>
            </Group>
          </Stack>
        )}
      </Modal>
    </>
  );
}
