import { fileTypeFromBuffer } from 'file-type';
import pdfParse from 'pdf-parse';
import mammoth from 'mammoth';
import * as crypto from 'crypto';
import { logger } from '../utils/logger';

export interface FileAnalysis {
  filename: string;
  fileType: string;
  mimeType: string;
  size: number;
  hash: {
    md5: string;
    sha1: string;
    sha256: string;
  };
  metadata: Record<string, any>;
  extractedText?: string;
  isExecutable: boolean;
  requiresSandbox: boolean;
}

export class FileAnalyzerService {
  async analyzeFile(fileBuffer: Buffer, filename: string): Promise<FileAnalysis> {
    const type = await fileTypeFromBuffer(fileBuffer);
    const mimeType = type?.mime || 'application/octet-stream';
    const fileTypeStr = type?.ext || this.getExtensionFromFilename(filename);
    
    // Calculate hashes
    const md5 = crypto.createHash('md5').update(fileBuffer).digest('hex');
    const sha1 = crypto.createHash('sha1').update(fileBuffer).digest('hex');
    const sha256 = crypto.createHash('sha256').update(fileBuffer).digest('hex');
    
    // Extract metadata and text based on file type
    let metadata: Record<string, any> = {};
    let extractedText: string | undefined;
    
    try {
      if (mimeType === 'application/pdf') {
        const pdfData = await pdfParse(fileBuffer);
        extractedText = pdfData.text;
        metadata = {
          pages: pdfData.numpages,
          info: pdfData.info,
          metadata: pdfData.metadata
        };
      } else if (
        mimeType.includes('wordprocessingml') || 
        mimeType.includes('msword') ||
        filename.toLowerCase().endsWith('.docx') ||
        filename.toLowerCase().endsWith('.doc')
      ) {
        const result = await mammoth.extractRawText({ buffer: fileBuffer });
        extractedText = result.value;
        metadata = {
          messages: result.messages
        };
      } else if (mimeType.includes('spreadsheetml') || filename.toLowerCase().endsWith('.xlsx')) {
        // Excel files - basic metadata only
        metadata = {
          type: 'spreadsheet'
        };
      }
    } catch (error: any) {
      logger.warn('Failed to extract text from file', { error: error.message, filename });
    }
    
    // Determine if file requires sandbox analysis
    const executableTypes = [
      'application/x-msdownload',
      'application/x-executable',
      'application/x-msdos-program',
      'application/x-dosexec',
      'application/x-elf',
      'application/x-sharedlib',
      'application/x-mach-binary',
      'application/vnd.microsoft.portable-executable'
    ];
    
    const executableExtensions = [
      '.exe', '.dll', '.bat', '.cmd', '.scr', '.com', '.pif',
      '.vbs', '.js', '.jar', '.msi', '.app', '.deb', '.rpm',
      '.sh', '.bin', '.run', '.elf', '.so', '.dylib'
    ];
    
    const scriptMimeTypes = [
      'application/javascript',
      'application/x-javascript',
      'text/javascript',
      'application/x-sh',
      'application/x-bash',
      'application/x-python',
      'text/x-python',
      'application/x-perl',
      'text/x-perl',
      'application/x-ruby',
      'text/x-ruby'
    ];
    
    const isExecutable = executableTypes.includes(mimeType) || 
                        executableExtensions.some(ext => 
                          filename.toLowerCase().endsWith(ext));
    
    const isScript = scriptMimeTypes.includes(mimeType) ||
                    ['.js', '.vbs', '.ps1', '.sh', '.bat', '.cmd', '.py', '.pl', '.rb'].some(ext =>
                      filename.toLowerCase().endsWith(ext));
    
    const requiresSandbox = isExecutable || isScript ||
                            mimeType.includes('javascript') ||
                            mimeType.includes('script') ||
                            mimeType === 'application/x-msdownload';
    
    return {
      filename,
      fileType: fileTypeStr,
      mimeType,
      size: fileBuffer.length,
      hash: { md5, sha1, sha256 },
      metadata,
      extractedText,
      isExecutable,
      requiresSandbox
    };
  }
  
  private getExtensionFromFilename(filename: string): string {
    const parts = filename.split('.');
    if (parts.length > 1) {
      return parts[parts.length - 1].toLowerCase();
    }
    return 'unknown';
  }
}
