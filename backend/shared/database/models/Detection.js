"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Detection = void 0;
const typeorm_1 = require("typeorm");
const Threat_1 = require("./Threat");
const Organization_1 = require("./Organization");
const DetectionFeedback_1 = require("./DetectionFeedback");
const IOCMatch_1 = require("./IOCMatch");
let Detection = class Detection {
};
exports.Detection = Detection;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], Detection.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid', nullable: true }),
    __metadata("design:type", String)
], Detection.prototype, "threat_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => Threat_1.Threat, (threat) => threat.detections, { onDelete: 'SET NULL', nullable: true }),
    (0, typeorm_1.JoinColumn)({ name: 'threat_id' }),
    __metadata("design:type", Threat_1.Threat)
], Detection.prototype, "threat", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid' }),
    __metadata("design:type", String)
], Detection.prototype, "organization_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => Organization_1.Organization, { onDelete: 'CASCADE' }),
    (0, typeorm_1.JoinColumn)({ name: 'organization_id' }),
    __metadata("design:type", Organization_1.Organization)
], Detection.prototype, "organization", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 50 }),
    __metadata("design:type", String)
], Detection.prototype, "detection_type", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 50, nullable: true }),
    __metadata("design:type", String)
], Detection.prototype, "model_version", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'jsonb' }),
    __metadata("design:type", Object)
], Detection.prototype, "input_data", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'jsonb' }),
    __metadata("design:type", Object)
], Detection.prototype, "analysis_result", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 5, scale: 2 }),
    __metadata("design:type", Number)
], Detection.prototype, "confidence_score", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'int', nullable: true }),
    __metadata("design:type", Number)
], Detection.prototype, "processing_time_ms", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' }),
    __metadata("design:type", Date)
], Detection.prototype, "detected_at", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], Detection.prototype, "created_at", void 0);
__decorate([
    (0, typeorm_1.OneToMany)(() => DetectionFeedback_1.DetectionFeedback, (feedback) => feedback.detection),
    __metadata("design:type", Array)
], Detection.prototype, "feedback", void 0);
__decorate([
    (0, typeorm_1.OneToMany)(() => IOCMatch_1.IOCMatch, (match) => match.detection),
    __metadata("design:type", Array)
], Detection.prototype, "ioc_matches", void 0);
exports.Detection = Detection = __decorate([
    (0, typeorm_1.Entity)('detections')
], Detection);
