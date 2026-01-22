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
exports.DetectionFeedback = void 0;
const typeorm_1 = require("typeorm");
const Detection_1 = require("./Detection");
const User_1 = require("./User");
let DetectionFeedback = class DetectionFeedback {
};
exports.DetectionFeedback = DetectionFeedback;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], DetectionFeedback.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid' }),
    __metadata("design:type", String)
], DetectionFeedback.prototype, "detection_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => Detection_1.Detection, (detection) => detection.feedback, { onDelete: 'CASCADE' }),
    (0, typeorm_1.JoinColumn)({ name: 'detection_id' }),
    __metadata("design:type", Detection_1.Detection)
], DetectionFeedback.prototype, "detection", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid', nullable: true }),
    __metadata("design:type", String)
], DetectionFeedback.prototype, "user_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => User_1.User, (user) => user.feedback, { onDelete: 'SET NULL', nullable: true }),
    (0, typeorm_1.JoinColumn)({ name: 'user_id' }),
    __metadata("design:type", User_1.User)
], DetectionFeedback.prototype, "user", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 20 }),
    __metadata("design:type", String)
], DetectionFeedback.prototype, "feedback_type", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'text', nullable: true }),
    __metadata("design:type", String)
], DetectionFeedback.prototype, "comment", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], DetectionFeedback.prototype, "created_at", void 0);
exports.DetectionFeedback = DetectionFeedback = __decorate([
    (0, typeorm_1.Entity)('detection_feedback')
], DetectionFeedback);
