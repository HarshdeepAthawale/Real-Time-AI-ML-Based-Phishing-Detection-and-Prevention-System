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
exports.ModelPerformance = void 0;
const typeorm_1 = require("typeorm");
const MLModel_1 = require("./MLModel");
let ModelPerformance = class ModelPerformance {
};
exports.ModelPerformance = ModelPerformance;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], ModelPerformance.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid' }),
    __metadata("design:type", String)
], ModelPerformance.prototype, "model_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => MLModel_1.MLModel, (model) => model.performance, { onDelete: 'CASCADE' }),
    (0, typeorm_1.JoinColumn)({ name: 'model_id' }),
    __metadata("design:type", MLModel_1.MLModel)
], ModelPerformance.prototype, "model", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'date' }),
    __metadata("design:type", Date)
], ModelPerformance.prototype, "date", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'int', default: 0 }),
    __metadata("design:type", Number)
], ModelPerformance.prototype, "total_predictions", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 10, scale: 2, nullable: true }),
    __metadata("design:type", Number)
], ModelPerformance.prototype, "avg_inference_time_ms", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 5, scale: 2, nullable: true }),
    __metadata("design:type", Number)
], ModelPerformance.prototype, "accuracy", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 5, scale: 2, nullable: true }),
    __metadata("design:type", Number)
], ModelPerformance.prototype, "precision", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 5, scale: 2, nullable: true }),
    __metadata("design:type", Number)
], ModelPerformance.prototype, "recall", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 5, scale: 2, nullable: true }),
    __metadata("design:type", Number)
], ModelPerformance.prototype, "f1_score", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 5, scale: 2, nullable: true }),
    __metadata("design:type", Number)
], ModelPerformance.prototype, "false_positive_rate", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], ModelPerformance.prototype, "created_at", void 0);
exports.ModelPerformance = ModelPerformance = __decorate([
    (0, typeorm_1.Entity)('model_performance'),
    (0, typeorm_1.Unique)(['model_id', 'date'])
], ModelPerformance);
