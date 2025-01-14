/* tslint:disable */
/* eslint-disable */
/**
 * Immich
 * Immich API
 *
 * The version of the OpenAPI document: 1.94.1
 * 
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


import * as runtime from '../runtime';
import type {
  MapTheme,
  SystemConfigDto,
  SystemConfigTemplateStorageOptionDto,
} from '../models/index';
import {
    MapThemeFromJSON,
    MapThemeToJSON,
    SystemConfigDtoFromJSON,
    SystemConfigDtoToJSON,
    SystemConfigTemplateStorageOptionDtoFromJSON,
    SystemConfigTemplateStorageOptionDtoToJSON,
} from '../models/index';

export interface GetMapStyleRequest {
    theme: MapTheme;
}

export interface UpdateConfigRequest {
    systemConfigDto: SystemConfigDto;
}

/**
 * 
 */
export class SystemConfigApi extends runtime.BaseAPI {

    /**
     */
    async getConfigRaw(initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<SystemConfigDto>> {
        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/system-config`,
            method: 'GET',
            headers: headerParameters,
            query: queryParameters,
        }, initOverrides);

        return new runtime.JSONApiResponse(response, (jsonValue) => SystemConfigDtoFromJSON(jsonValue));
    }

    /**
     */
    async getConfig(initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<SystemConfigDto> {
        const response = await this.getConfigRaw(initOverrides);
        return await response.value();
    }

    /**
     */
    async getConfigDefaultsRaw(initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<SystemConfigDto>> {
        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/system-config/defaults`,
            method: 'GET',
            headers: headerParameters,
            query: queryParameters,
        }, initOverrides);

        return new runtime.JSONApiResponse(response, (jsonValue) => SystemConfigDtoFromJSON(jsonValue));
    }

    /**
     */
    async getConfigDefaults(initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<SystemConfigDto> {
        const response = await this.getConfigDefaultsRaw(initOverrides);
        return await response.value();
    }

    /**
     */
    async getMapStyleRaw(requestParameters: GetMapStyleRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<object>> {
        if (requestParameters.theme === null || requestParameters.theme === undefined) {
            throw new runtime.RequiredError('theme','Required parameter requestParameters.theme was null or undefined when calling getMapStyle.');
        }

        const queryParameters: any = {};

        if (requestParameters.theme !== undefined) {
            queryParameters['theme'] = requestParameters.theme;
        }

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/system-config/map/style.json`,
            method: 'GET',
            headers: headerParameters,
            query: queryParameters,
        }, initOverrides);

        return new runtime.JSONApiResponse<any>(response);
    }

    /**
     */
    async getMapStyle(requestParameters: GetMapStyleRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<object> {
        const response = await this.getMapStyleRaw(requestParameters, initOverrides);
        return await response.value();
    }

    /**
     */
    async getStorageTemplateOptionsRaw(initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<SystemConfigTemplateStorageOptionDto>> {
        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/system-config/storage-template-options`,
            method: 'GET',
            headers: headerParameters,
            query: queryParameters,
        }, initOverrides);

        return new runtime.JSONApiResponse(response, (jsonValue) => SystemConfigTemplateStorageOptionDtoFromJSON(jsonValue));
    }

    /**
     */
    async getStorageTemplateOptions(initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<SystemConfigTemplateStorageOptionDto> {
        const response = await this.getStorageTemplateOptionsRaw(initOverrides);
        return await response.value();
    }

    /**
     */
    async updateConfigRaw(requestParameters: UpdateConfigRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<runtime.ApiResponse<SystemConfigDto>> {
        if (requestParameters.systemConfigDto === null || requestParameters.systemConfigDto === undefined) {
            throw new runtime.RequiredError('systemConfigDto','Required parameter requestParameters.systemConfigDto was null or undefined when calling updateConfig.');
        }

        const queryParameters: any = {};

        const headerParameters: runtime.HTTPHeaders = {};

        headerParameters['Content-Type'] = 'application/json';

        if (this.configuration && this.configuration.apiKey) {
            headerParameters["x-api-key"] = this.configuration.apiKey("x-api-key"); // api_key authentication
        }

        if (this.configuration && this.configuration.accessToken) {
            const token = this.configuration.accessToken;
            const tokenString = await token("bearer", []);

            if (tokenString) {
                headerParameters["Authorization"] = `Bearer ${tokenString}`;
            }
        }
        const response = await this.request({
            path: `/system-config`,
            method: 'PUT',
            headers: headerParameters,
            query: queryParameters,
            body: SystemConfigDtoToJSON(requestParameters.systemConfigDto),
        }, initOverrides);

        return new runtime.JSONApiResponse(response, (jsonValue) => SystemConfigDtoFromJSON(jsonValue));
    }

    /**
     */
    async updateConfig(requestParameters: UpdateConfigRequest, initOverrides?: RequestInit | runtime.InitOverrideFunction): Promise<SystemConfigDto> {
        const response = await this.updateConfigRaw(requestParameters, initOverrides);
        return await response.value();
    }

}
