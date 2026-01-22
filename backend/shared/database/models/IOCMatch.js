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
exports.IOCMatch = void 0;
const typeorm_1 = require("typeorm");
const IOC_1 = require("./IOC");
const Detection_1 = require("./Detection");
let IOCMatch = class IOCMatch {
};
exports.IOCMatch = IOCMatch;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], IOCMatch.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid' }),
    __metadata("design:type", String)
], IOCMatch.prototype, "ioc_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => IOC_1.IOC, (ioc) => ioc.matches, { onDelete: 'CASCADE' }),
    (0, typeorm_1.JoinColumn)({ name: 'ioc_id' }),
    __metadata("design:type", IOC_1.IOC)
], IOCMatch.prototype, "ioc", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid' }),
    __metadata("design:type", String)
], IOCMatch.prototype, "detection_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => Detection_1.Detection, (detection) => detection.ioc_matches, { onDelete: 'CASCADE' }),
    (0, typeorm_1.JoinColumn)({ name: 'detection_id' }),
    __metadata("design:type", Detection_1.Detection)
], IOCMatch.prototype, "detection", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' }),
    __metadata("design:type", Date)
], IOCMatch.prototype, "matched_at", void 0);
exports.IOCMatch = IOCMatch = __decorate([
    (0, typeorm_1.Entity)('ioc_matches')
], IOCMatch);
