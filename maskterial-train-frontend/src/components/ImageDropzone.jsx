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
  ...props 
}) {
  const [isUploading, setIsUploading] = useState(false);

  const handleDrop = async (files) => {
    if (files.length === 0) return;

    const file = files[0];

    // If autoUpload is enabled, upload to server
    if (autoUpload) {
      setIsUploading(true);
      
      try {
        const result = await uploadImage(file, file.name);
        
        if (showNotifications) {
          notifications.show({
            title: "Upload Successful",
            message: `${file.name} uploaded successfully`,
            color: "green",
            autoClose: 3000,
          });
        }
        
        // Call success callback if provided
        if (onUploadSuccess) {
          onUploadSuccess(result, file);
        }
        
        // Still call the original handler with the file
        if (handleImageUpload) {
          handleImageUpload(files, result);
        }
        
      } catch (error) {
        console.error("Upload error:", error);
        
        if (showNotifications) {
          notifications.show({
            title: "Upload Failed",
            message: error.message || "Failed to upload image",
            color: "red",
            autoClose: 5000,
          });
        }
        
        // Call error callback if provided
        if (onUploadError) {
          onUploadError(error, file);
        }
      } finally {
        setIsUploading(false);
      }
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
                  Uploading image...
                </Text>
              </Group>
            </>
          ) : (
            <>
              <Text size="xl" inline>
                Drag a flake image here or click to select files
              </Text>
              <Text size="sm" c="dimmed" inline mt={7}>
                The image should be a .jpg, .jpeg, or .png file
              </Text>
            </>
          )}
        </div>
      </Group>
    </Dropzone>
  );
}
