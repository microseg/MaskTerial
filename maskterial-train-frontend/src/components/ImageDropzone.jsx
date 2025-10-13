import { useState } from "react";
import { Group, Text, Loader } from "@mantine/core";
import { Dropzone, IMAGE_MIME_TYPE } from "@mantine/dropzone";
import { notifications } from "@mantine/notifications";
import { uploadImage } from "../utils/apiClient";
import styles from "./ImageDropzone.module.css";

export function ImageDropzone({ 
  handleImageUpload, 
  autoUpload = false, 
  onUploadSuccess = null,
  onUploadError = null,
  showNotifications = true,
  multiple = true,  // 默认支持多选
  ...props 
}) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({ current: 0, total: 0 });

  const handleDrop = async (files) => {
    if (files.length === 0) return;

    // If autoUpload is enabled, upload to server
    if (autoUpload) {
      setIsUploading(true);
      setUploadProgress({ current: 0, total: files.length });
      
      const results = [];
      const errors = [];
      
      // 批量上传所有文件
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        setUploadProgress({ current: i + 1, total: files.length });
        
        try {
          const result = await uploadImage(file, file.name);
          results.push({ file, result });
          
          console.log(`Uploaded ${i + 1}/${files.length}: ${file.name}`);
          
        } catch (error) {
          console.error(`Upload error for ${file.name}:`, error);
          errors.push({ file, error });
        }
      }
      
      // 显示批量上传结果通知
      if (showNotifications) {
        if (errors.length === 0) {
          notifications.show({
            title: "Upload Successful",
            message: `${results.length} image${results.length > 1 ? 's' : ''} uploaded successfully`,
            color: "green",
            autoClose: 3000,
          });
        } else if (results.length > 0) {
          notifications.show({
            title: "Partial Success",
            message: `${results.length} uploaded, ${errors.length} failed`,
            color: "orange",
            autoClose: 5000,
          });
        } else {
          notifications.show({
            title: "Upload Failed",
            message: `Failed to upload ${errors.length} image${errors.length > 1 ? 's' : ''}`,
            color: "red",
            autoClose: 5000,
          });
        }
      }
      
      // Call success callback for each successful upload
      if (onUploadSuccess && results.length > 0) {
        results.forEach(({ result, file }) => {
          onUploadSuccess(result, file);
        });
      }
      
      // Call error callback for each failed upload
      if (onUploadError && errors.length > 0) {
        errors.forEach(({ error, file }) => {
          onUploadError(error, file);
        });
      }
      
      // Call the original handler with all files and results
      if (handleImageUpload && results.length > 0) {
        // 为了兼容，只传第一个文件和结果给handleImageUpload
        // 但通过onUploadSuccess可以处理所有文件
        const firstResult = results[0];
        handleImageUpload([firstResult.file], firstResult.result);
      }
      
      setIsUploading(false);
      setUploadProgress({ current: 0, total: 0 });
      
    } else {
      // Just call the handler without uploading
      if (handleImageUpload) {
        handleImageUpload(files);
      }
    }
  };

  return (
    <Dropzone
      onDrop={handleDrop}
      onReject={(files) => console.log("rejected files", files)}
      maxSize={5 * 1024 ** 2}
      accept={IMAGE_MIME_TYPE}
      disabled={isUploading}
      multiple={multiple}
      {...props}
    >
      <Group
        justify="center"
        gap="xl"
        style={{ pointerEvents: "none" }}
        className={styles.dropzoneGroup}
      >
        <Dropzone.Accept></Dropzone.Accept>
        <Dropzone.Reject></Dropzone.Reject>
        <Dropzone.Idle></Dropzone.Idle>

        <div>
          {isUploading ? (
            <>
              <Group gap="sm" justify="center">
                <Loader size="sm" />
                <Text size="xl" inline>
                  {uploadProgress.total > 1 
                    ? `Uploading ${uploadProgress.current}/${uploadProgress.total} images...`
                    : 'Uploading image...'
                  }
                </Text>
              </Group>
            </>
          ) : (
            <>
              <Text size="xl" inline>
                {multiple 
                  ? 'Drag images here or click to select multiple files'
                  : 'Drag a flake image here or click to select files'
                }
              </Text>
              <Text size="sm" c="dimmed" inline mt={7}>
                {multiple
                  ? 'You can select multiple .jpg, .jpeg, or .png files'
                  : 'The image should be a .jpg, .jpeg, or .png file'
                }
              </Text>
            </>
          )}
        </div>
      </Group>
    </Dropzone>
  );
}
