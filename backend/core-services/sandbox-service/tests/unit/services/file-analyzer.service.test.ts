import 'reflect-metadata';
import { FileAnalyzerService } from '../../../src/services/file-analyzer.service';
import { mockFileBuffer } from '../../fixtures/mock-data';

jest.mock('file-type');
jest.mock('pdf-parse');
jest.mock('mammoth');

describe('FileAnalyzerService', () => {
  let service: FileAnalyzerService;

  beforeEach(() => {
    service = new FileAnalyzerService();
  });

  describe('analyzeFile', () => {
    it('should analyze an executable file correctly', async () => {
      const { fileTypeFromBuffer } = require('file-type');
      fileTypeFromBuffer.mockResolvedValue({
        mime: 'application/x-msdownload',
        ext: 'exe',
      });

      const result = await service.analyzeFile(mockFileBuffer, 'test.exe');

      expect(result.filename).toBe('test.exe');
      expect(result.fileType).toBe('exe');
      expect(result.mimeType).toBe('application/x-msdownload');
      expect(result.isExecutable).toBe(true);
      expect(result.requiresSandbox).toBe(true);
      expect(result.hash).toHaveProperty('md5');
      expect(result.hash).toHaveProperty('sha1');
      expect(result.hash).toHaveProperty('sha256');
    });

    it('should analyze a PDF file and extract text', async () => {
      const { fileTypeFromBuffer } = require('file-type');
      const pdfParse = require('pdf-parse');
      
      fileTypeFromBuffer.mockResolvedValue({
        mime: 'application/pdf',
        ext: 'pdf',
      });

      pdfParse.mockResolvedValue({
        text: 'Sample PDF text',
        numpages: 1,
        info: {},
      });

      const result = await service.analyzeFile(mockFileBuffer, 'test.pdf');

      expect(result.fileType).toBe('pdf');
      expect(result.mimeType).toBe('application/pdf');
      expect(result.extractedText).toBe('Sample PDF text');
      expect(result.metadata.pages).toBe(1);
      expect(result.requiresSandbox).toBe(false);
    });

    it('should analyze a Word document and extract text', async () => {
      const { fileTypeFromBuffer } = require('file-type');
      const mammoth = require('mammoth');
      
      fileTypeFromBuffer.mockResolvedValue({
        mime: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        ext: 'docx',
      });

      mammoth.extractRawText.mockResolvedValue({
        value: 'Sample Word text',
        messages: [],
      });

      const result = await service.analyzeFile(mockFileBuffer, 'test.docx');

      expect(result.fileType).toBe('docx');
      expect(result.extractedText).toBe('Sample Word text');
      expect(result.requiresSandbox).toBe(false);
    });

    it('should identify JavaScript files as requiring sandbox', async () => {
      const { fileTypeFromBuffer } = require('file-type');
      
      fileTypeFromBuffer.mockResolvedValue({
        mime: 'application/javascript',
        ext: 'js',
      });

      const result = await service.analyzeFile(mockFileBuffer, 'test.js');

      expect(result.requiresSandbox).toBe(true);
    });

    it('should handle files without detectable type', async () => {
      const { fileTypeFromBuffer } = require('file-type');
      
      fileTypeFromBuffer.mockResolvedValue(null);

      const result = await service.analyzeFile(mockFileBuffer, 'test.unknown');

      expect(result.fileType).toBe('unknown');
      expect(result.mimeType).toBe('application/octet-stream');
    });

    it('should calculate correct hashes', async () => {
      const { fileTypeFromBuffer } = require('file-type');
      
      fileTypeFromBuffer.mockResolvedValue({
        mime: 'text/plain',
        ext: 'txt',
      });

      const result = await service.analyzeFile(mockFileBuffer, 'test.txt');

      expect(result.hash.md5).toHaveLength(32);
      expect(result.hash.sha1).toHaveLength(40);
      expect(result.hash.sha256).toHaveLength(64);
    });
  });
});
