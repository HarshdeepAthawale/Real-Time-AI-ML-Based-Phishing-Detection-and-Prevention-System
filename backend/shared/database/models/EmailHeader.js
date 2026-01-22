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
exports.EmailHeader = void 0;
const typeorm_1 = require("typeorm");
const EmailMessage_1 = require("./EmailMessage");
let EmailHeader = class EmailHeader {
};
exports.EmailHeader = EmailHeader;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], EmailHeader.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid' }),
    __metadata("design:type", String)
], EmailHeader.prototype, "email_message_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => EmailMessage_1.EmailMessage, (email) => email.headers, { onDelete: 'CASCADE' }),
    (0, typeorm_1.JoinColumn)({ name: 'email_message_id' }),
    __metadata("design:type", EmailMessage_1.EmailMessage)
], EmailHeader.prototype, "email_message", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 255 }),
    __metadata("design:type", String)
], EmailHeader.prototype, "header_name", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'text', nullable: true }),
    __metadata("design:type", String)
], EmailHeader.prototype, "header_value", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], EmailHeader.prototype, "created_at", void 0);
exports.EmailHeader = EmailHeader = __decorate([
    (0, typeorm_1.Entity)('email_headers')
], EmailHeader);
