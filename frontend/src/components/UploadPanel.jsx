export const UploadPanel = ({
  isDragging,
  handleDragEnter,
  handleDragLeave,
  handleDragOver,
  handleDrop,
  fileInputRef,
  handleFileInput
}) => {
  return (
    <div className="left-panel">
      <div className="upload-wrapper">
        <div className="bone-frame">
          <div
            className={`upload-dropzone ${isDragging ? 'drag-active' : ''}`}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            <svg className="upload-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            <p className="upload-text">Upload Scan</p>
            <p className="upload-subtext">Drop your X-ray or browse</p>

            <div className="upload-btn-fake">Browse Files</div>

            <input
              type="file"
              multiple
              accept="image/*"
              className="file-input"
              ref={fileInputRef}
              onChange={handleFileInput}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
